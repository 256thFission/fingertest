#!/usr/bin/env python3
"""
Phase 1: The "Janitor" Data Pipeline
Streams Discord JSON dumps and creates cleaned, stratified context blocks.
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict
from typing import Iterator, Dict, List, Tuple
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


class DiscordDataCleaner:
    """Streaming Discord JSON cleaner with bot/system filtering."""

    def __init__(self):
        # Regex patterns for normalization
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.mention_pattern = re.compile(r"<@!?\d+>")
        self.command_pattern = re.compile(r"^[/!;]")
        self.code_block_pattern = re.compile(r"```")

    def is_bot(self, msg_data: dict) -> bool:
        """Check if message is from a bot."""
        author = msg_data.get("author", {})
        if author.get("bot", False):
            return True
        if author.get("discriminator") == "0000":
            return True
        return False

    def is_system_message(self, content: str) -> bool:
        """Check if message is a system command or code block."""
        if not content or not content.strip():
            return True
        if self.command_pattern.match(content.strip()):
            return True
        if self.code_block_pattern.search(content):
            return True
        return False

    def normalize_content(self, content: str) -> str:
        """Normalize URLs and mentions, keep emojis."""
        content = self.url_pattern.sub("[URL]", content)
        content = self.mention_pattern.sub("[USER]", content)
        # Normalize whitespace but keep structure
        content = re.sub(r"\s+", " ", content).strip()
        return content

    def parse_message(
        self, msg_data: dict, server_id: str = "unknown"
    ) -> Message | None:
        """Parse raw Discord JSON to Message object."""
        try:
            # Skip bots
            if self.is_bot(msg_data):
                return None

            content = msg_data.get("content", "")

            # Skip system messages
            if self.is_system_message(content):
                return None

            # Normalize content
            content = self.normalize_content(content)

            # Skip if nothing left after normalization
            if len(content.strip()) < 3:
                return None

            author = msg_data.get("author", {})
            return Message(
                author_id=str(author.get("id", "unknown")),
                author_name=author.get("username", "unknown"),
                channel_id=str(msg_data.get("channel_id", "unknown")),
                server_id=server_id,
                content=content,
                timestamp=msg_data.get("timestamp", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to parse message: {e}")
            return None


class SessionAggregator:
    """Aggregates messages into context blocks using sliding window."""

    def __init__(
        self,
        max_tokens: int = 512,
        min_tokens: int = 20,
        time_window_seconds: int = 300,
    ):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.time_window_seconds = time_window_seconds

    def estimate_tokens(self, text: str) -> int:
        """Fast token estimation (roughly 4 chars per token for English)."""
        return len(text) // 4

    def parse_timestamp(self, ts: str) -> float:
        """Parse Discord timestamp to Unix time."""
        from datetime import datetime

        try:
            # Discord uses ISO 8601 format
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.timestamp()
        except:
            return 0.0

    def aggregate_messages(self, messages: List[Message]) -> List[Dict]:
        """
        Group consecutive messages from (User, Channel) pairs into context blocks.
        Logic: Group if time_delta < 5 minutes.
        """
        if not messages:
            return []

        # Sort by timestamp
        messages = sorted(messages, key=lambda m: m.timestamp)

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
                    if time_delta > self.time_window_seconds:
                        should_split = True

                # Check if adding this would exceed max_tokens
                if current_tokens + msg_tokens > self.max_tokens:
                    should_split = True

                if should_split and current_block:
                    # Save current block if it meets minimum
                    block_text = " ".join(current_block)
                    if self.estimate_tokens(block_text) >= self.min_tokens:
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
                if self.estimate_tokens(block_text) >= self.min_tokens:
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


class DiscordDataPipeline:
    """Main pipeline for processing Discord data dumps."""

    def __init__(self, raw_data_dir: str, output_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cleaner = DiscordDataCleaner()
        self.aggregator = SessionAggregator()

    def stream_json_files(self) -> Iterator[Tuple[dict, str]]:
        """Stream JSON files from raw data directory."""
        json_files = list(self.raw_data_dir.glob("**/*.json"))

        logger.info(f"Found {len(json_files)} JSON files")

        for json_file in json_files:
            server_id = json_file.parent.name
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Handle different Discord export formats
                    messages = (
                        data if isinstance(data, list) else data.get("messages", [])
                    )

                    for msg in messages:
                        yield msg, server_id

            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
                continue

    def process_streaming(self, chunk_size: int = 10000) -> List[Dict]:
        """Process data in streaming fashion to avoid OOM."""
        all_blocks = []
        message_buffer = []
        total_messages = 0
        total_valid = 0

        logger.info("Starting streaming data processing...")

        for msg_data, server_id in tqdm(
            self.stream_json_files(), desc="Processing messages"
        ):
            total_messages += 1

            msg = self.cleaner.parse_message(msg_data, server_id)
            if msg:
                message_buffer.append(msg)
                total_valid += 1

            # Process in chunks to manage memory
            if len(message_buffer) >= chunk_size:
                blocks = self.aggregator.aggregate_messages(message_buffer)
                all_blocks.extend(blocks)
                logger.info(
                    f"Processed {total_messages} messages, {len(all_blocks)} blocks so far"
                )
                message_buffer = []

        # Process remaining messages
        if message_buffer:
            blocks = self.aggregator.aggregate_messages(message_buffer)
            all_blocks.extend(blocks)

        logger.info(
            f"Total messages: {total_messages}, Valid: {total_valid}, Blocks: {len(all_blocks)}"
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

        logger.info(f"Valid authors: {len(valid_authors)}")

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

        logger.info(f"Train blocks: {len(train_data)}")
        logger.info(f"Val blocks: {len(val_data)}")
        logger.info(f"Test blocks: {len(test_data)}")

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

    parser = argparse.ArgumentParser(description="Discord Authorship Data Preprocessor")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw Discord JSON dumps",
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

    args = parser.parse_args()

    # Check if raw data exists
    if not Path(args.raw_dir).exists():
        logger.error(f"Raw data directory not found: {args.raw_dir}")
        logger.info("Please place Discord JSON dumps in data/raw/")
        logger.info("Expected structure: data/raw/server_name/*.json")
        return

    # Initialize pipeline
    pipeline = DiscordDataPipeline(args.raw_dir, args.output_dir)

    # Process data
    logger.info("=" * 80)
    logger.info("Phase 1: Data Preprocessing")
    logger.info("=" * 80)

    blocks = pipeline.process_streaming()

    if not blocks:
        logger.error("No valid blocks generated. Check your data format.")
        return

    # Stratify
    splits = pipeline.stratify_data(blocks, min_blocks_per_author=args.min_blocks)

    # Save
    pipeline.save_to_parquet(splits)

    logger.info("=" * 80)
    logger.info("Preprocessing complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
