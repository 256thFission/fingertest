#!/usr/bin/env python3
"""
Preprocessor for Existing Parquet Data
Adapts existing server parquet files to the format expected by training scripts.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Iterator, Dict, List
from dataclasses import dataclass
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Structured message representation."""

    author_id: str
    author_name: str
    channel_id: str
    server_id: str
    content: str
    timestamp: str


class ParquetDataProcessor:
    """Processes existing parquet files and creates training datasets."""

    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "data/processed",
        skip_channel_mapping: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_channel_mapping = skip_channel_mapping

        # Cache for channel_id mapping (message_id -> channel_id)
        self.channel_cache = {}

    def load_channel_mappings(
        self, server_id: str, limit_lines: int = None
    ) -> Dict[str, str]:
        """
        Load channel_id mappings from JSON file for a given server.
        Returns dict: message_id -> channel_id

        For very large files, this can be skipped by setting limit_lines=0.
        """
        if limit_lines == 0:
            logger.info(f"Skipping channel mapping load for server {server_id}")
            return {}

        json_file = self.data_dir / f"{server_id}.json"

        if not json_file.exists():
            logger.warning(
                f"JSON file not found for server {server_id}, channel_id will be 'unknown'"
            )
            return {}

        logger.info(f"Loading channel mappings from {json_file}")
        channel_map = {}
        line_count = 0

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                for line in f:
                    if limit_lines and line_count >= limit_lines:
                        logger.info(f"Reached limit of {limit_lines:,} lines")
                        break

                    try:
                        msg = json.loads(line.strip())
                        msg_id = msg.get("id")
                        channel_id = msg.get("channel_id")
                        if msg_id and channel_id:
                            channel_map[msg_id] = channel_id
                    except json.JSONDecodeError:
                        pass

                    line_count += 1
                    if line_count % 100000 == 0:
                        logger.info(
                            f"Loaded {line_count:,} lines, {len(channel_map):,} mappings..."
                        )

        except Exception as e:
            logger.error(f"Error loading channel mappings: {e}")

        logger.info(
            f"Loaded {len(channel_map):,} channel mappings from {line_count:,} lines"
        )
        return channel_map

    def stream_parquet_messages(self) -> Iterator[Message]:
        """Stream messages from all parquet directories."""
        parquet_dirs = sorted(self.data_dir.glob("*.parquet"))

        logger.info(f"Found {len(parquet_dirs)} parquet directories")

        for parquet_dir in parquet_dirs:
            server_id = parquet_dir.stem
            logger.info(f"Processing server: {server_id}")

            # Load channel mappings for this server (or skip for speed)
            if self.skip_channel_mapping:
                channel_map = {}
                logger.info("Skipping channel mapping (using server_id as fallback)")
            else:
                channel_map = self.load_channel_mappings(server_id, limit_lines=500000)

            # Read all parquet files in directory
            parquet_files = list(parquet_dir.glob("*.parquet"))
            logger.info(f"Found {len(parquet_files)} parquet files")

            for parquet_file in parquet_files:
                try:
                    table = pq.read_table(parquet_file)
                    df = table.to_pandas()

                    logger.info(
                        f"Processing {len(df):,} messages from {parquet_file.name}"
                    )

                    for _, row in df.iterrows():
                        # Skip bots
                        if row.get("is_bot", False):
                            continue

                        # Skip empty content
                        content = row.get("content", "").strip()
                        if not content or len(content) < 3:
                            continue

                        # Get channel_id from cache
                        message_id = row.get("message_id", "")
                        channel_id = channel_map.get(message_id, "unknown")

                        yield Message(
                            author_id=str(row.get("user_id", "unknown")),
                            author_name=str(row.get("username", "unknown")),
                            channel_id=str(channel_id),
                            server_id=server_id,
                            content=content,
                            timestamp=str(row.get("timestamp", "")),
                        )

                except Exception as e:
                    logger.error(f"Error processing {parquet_file}: {e}")
                    continue

    def estimate_tokens(self, text: str) -> int:
        """Fast token estimation (roughly 4 chars per token for English)."""
        return len(text) // 4

    def parse_timestamp(self, ts: str) -> float:
        """Parse timestamp to Unix time."""
        from datetime import datetime

        try:
            # Handle pandas Timestamp objects
            if hasattr(ts, "timestamp"):
                return ts.timestamp()
            # Try ISO format
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            return dt.timestamp()
        except:
            return 0.0

    def aggregate_messages(
        self,
        messages: List[Message],
        max_tokens: int = 512,
        min_tokens: int = 20,
        time_window_seconds: int = 300,
    ) -> List[Dict]:
        """
        Group consecutive messages from (User, Channel) pairs into context blocks.
        Logic: Group if time_delta < 5 minutes.
        """
        if not messages:
            return []

        # Sort by timestamp
        messages = sorted(messages, key=lambda m: self.parse_timestamp(m.timestamp))

        # Group by (author_id, channel_id)
        user_channel_messages = defaultdict(list)
        for msg in messages:
            key = (msg.author_id, msg.channel_id)
            user_channel_messages[key].append(msg)

        context_blocks = []

        for (author_id, channel_id), msgs in user_channel_messages.items():
            if not msgs:
                continue

            current_block = []
            current_tokens = 0
            last_timestamp = None

            for msg in msgs:
                msg_tokens = self.estimate_tokens(msg.content)
                msg_time = self.parse_timestamp(msg.timestamp)

                # Check if we should start a new block
                should_split = False
                if last_timestamp is not None:
                    time_delta = abs(msg_time - last_timestamp)
                    if time_delta > time_window_seconds:
                        should_split = True

                # Check if adding this would exceed max_tokens
                if current_tokens + msg_tokens > max_tokens:
                    should_split = True

                if should_split and current_block:
                    # Save current block if it meets minimum
                    block_text = " ".join(current_block)
                    if self.estimate_tokens(block_text) >= min_tokens:
                        context_blocks.append(
                            {
                                "author_id": author_id,
                                "channel_id": channel_id,
                                "server_id": msgs[0].server_id,
                                "text": block_text,
                                "num_messages": len(current_block),
                            }
                        )
                    current_block = []
                    current_tokens = 0

                current_block.append(msg.content)
                current_tokens += msg_tokens
                last_timestamp = msg_time

            # Don't forget the last block
            if current_block:
                block_text = " ".join(current_block)
                if self.estimate_tokens(block_text) >= min_tokens:
                    context_blocks.append(
                        {
                            "author_id": author_id,
                            "channel_id": channel_id,
                            "server_id": msgs[0].server_id,
                            "text": block_text,
                            "num_messages": len(current_block),
                        }
                    )

        return context_blocks

    def process_streaming(self, chunk_size: int = 50000) -> List[Dict]:
        """Process data in streaming fashion to avoid OOM."""
        all_blocks = []
        message_buffer = []
        total_messages = 0
        total_valid = 0

        logger.info("Starting streaming data processing...")

        for msg in tqdm(self.stream_parquet_messages(), desc="Processing messages"):
            total_messages += 1
            message_buffer.append(msg)
            total_valid += 1

            # Process in chunks to manage memory
            if len(message_buffer) >= chunk_size:
                blocks = self.aggregate_messages(message_buffer)
                all_blocks.extend(blocks)
                logger.info(
                    f"Processed {total_messages:,} messages, {len(all_blocks):,} blocks so far"
                )
                message_buffer = []

        # Process remaining messages
        if message_buffer:
            blocks = self.aggregate_messages(message_buffer)
            all_blocks.extend(blocks)

        logger.info(
            f"Total messages: {total_messages:,}, Valid: {total_valid:,}, Blocks: {len(all_blocks):,}"
        )
        return all_blocks

    def stratify_data(
        self, blocks: List[Dict], min_blocks_per_author: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Stratify data into train/val/test splits.

        Train: Users with >= min_blocks_per_author (90% of their data)
        Validation: Remaining 10% from train users
        Test (Zero-Shot): Bottom 1000 valid authors (never seen during training)
        """
        # Count blocks per author
        author_blocks = defaultdict(list)
        for block in blocks:
            author_blocks[block["author_id"]].append(block)

        # Filter authors with sufficient data
        valid_authors = {
            author_id: blks
            for author_id, blks in author_blocks.items()
            if len(blks) >= min_blocks_per_author
        }

        logger.info(f"Valid authors: {len(valid_authors):,}")

        # Sort by block count
        sorted_authors = sorted(valid_authors.items(), key=lambda x: len(x[1]))

        # Reserve bottom 1000 for test (zero-shot)
        test_size = min(1000, len(sorted_authors) // 10)
        test_authors = sorted_authors[:test_size]
        train_val_authors = sorted_authors[test_size:]

        logger.info(f"Test authors (zero-shot): {len(test_authors)}")
        logger.info(f"Train+Val authors: {len(train_val_authors)}")

        # Split train/val (90/10)
        train_data = []
        val_data = []
        test_data = []

        # Test set: all blocks from test authors
        for author_id, blks in test_authors:
            test_data.extend(blks)

        # Train/Val split
        for author_id, blks in train_val_authors:
            split_idx = int(len(blks) * 0.9)
            train_data.extend(blks[:split_idx])
            val_data.extend(blks[split_idx:])

        logger.info(f"Train blocks: {len(train_data):,}")
        logger.info(f"Val blocks: {len(val_data):,}")
        logger.info(f"Test blocks: {len(test_data):,}")

        return {"train": train_data, "val": val_data, "test": test_data}

    def save_to_parquet(self, splits: Dict[str, List[Dict]]):
        """Save splits to Parquet files for efficient loading."""
        for split_name, data in splits.items():
            if not data:
                logger.warning(f"No data for split: {split_name}")
                continue

            # Convert to PyArrow Table
            schema = pa.schema(
                [
                    ("author_id", pa.string()),
                    ("channel_id", pa.string()),
                    ("server_id", pa.string()),
                    ("text", pa.string()),
                    ("num_messages", pa.int64()),
                ]
            )

            table = pa.Table.from_pylist(data, schema=schema)

            output_path = self.output_dir / f"{split_name}.parquet"
            pq.write_table(table, output_path, compression="snappy")

            logger.info(f"Saved {split_name} split to {output_path}")


def main():
    """Main entry point for preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(description="Process existing Parquet files")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing {server_id}.parquet directories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--min-blocks", type=int, default=5, help="Minimum blocks per author to include"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50000, help="Messages per processing chunk"
    )
    parser.add_argument(
        "--skip-channel-mapping",
        action="store_true",
        help="Skip loading channel mappings from JSON (faster, uses server_id as channel_id)",
    )

    args = parser.parse_args()

    # Check if data exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return

    parquet_dirs = list(data_path.glob("*.parquet"))
    if not parquet_dirs:
        logger.error(f"No .parquet directories found in {args.data_dir}")
        return

    # Initialize pipeline
    processor = ParquetDataProcessor(
        args.data_dir, args.output_dir, skip_channel_mapping=args.skip_channel_mapping
    )

    # Process data
    logger.info("=" * 80)
    logger.info("Parquet Data Preprocessing")
    logger.info("=" * 80)

    blocks = processor.process_streaming(chunk_size=args.chunk_size)

    if not blocks:
        logger.error("No valid blocks generated. Check your data format.")
        return

    # Stratify
    splits = processor.stratify_data(blocks, min_blocks_per_author=args.min_blocks)

    # Save
    processor.save_to_parquet(splits)

    logger.info("=" * 80)
    logger.info("Preprocessing complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
