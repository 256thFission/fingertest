#!/usr/bin/env python3
"""
Generate synthetic Discord JSON data for testing.
This creates realistic-looking Discord messages with varying writing styles.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import string

# Writing style templates
WRITING_STYLES = {
    "formal": {
        "vocabulary": [
            "indeed",
            "therefore",
            "furthermore",
            "nevertheless",
            "consequently",
        ],
        "punctuation_rate": 0.9,
        "capitalization": True,
        "emoji_rate": 0.05,
        "abbrev_rate": 0.1,
    },
    "casual": {
        "vocabulary": ["yeah", "nah", "like", "kinda", "totally"],
        "punctuation_rate": 0.5,
        "capitalization": False,
        "emoji_rate": 0.3,
        "abbrev_rate": 0.4,
    },
    "technical": {
        "vocabulary": ["implement", "configure", "optimize", "refactor", "deploy"],
        "punctuation_rate": 0.8,
        "capitalization": True,
        "emoji_rate": 0.1,
        "abbrev_rate": 0.3,
    },
    "enthusiastic": {
        "vocabulary": ["amazing", "awesome", "fantastic", "incredible", "wow"],
        "punctuation_rate": 0.7,
        "capitalization": True,
        "emoji_rate": 0.6,
        "abbrev_rate": 0.2,
    },
    "terse": {
        "vocabulary": ["ok", "sure", "yep", "nope", "fine"],
        "punctuation_rate": 0.3,
        "capitalization": False,
        "emoji_rate": 0.1,
        "abbrev_rate": 0.6,
    },
}

EMOJIS = ["😀", "😂", "👍", "❤️", "🔥", "✨", "🎉", "👀", "🤔", "😎"]

BASE_MESSAGES = [
    "what do you think about this",
    "i was working on the project today",
    "has anyone tried the new feature",
    "looking forward to the next update",
    "thanks for the help everyone",
    "can someone explain how this works",
    "just finished reading the documentation",
    "this is really interesting",
    "anyone available for a quick chat",
    "great work on the recent changes",
]


def generate_message(style: dict, base: str) -> str:
    """Generate a message with a specific writing style."""
    tokens = base.split()

    # Add style-specific vocabulary
    if random.random() < 0.3:
        insert_pos = random.randint(0, len(tokens))
        tokens.insert(insert_pos, random.choice(style["vocabulary"]))

    # Apply abbreviations
    if random.random() < style["abbrev_rate"]:
        abbrevs = {"you": "u", "are": "r", "to": "2", "for": "4", "be": "b"}
        tokens = [abbrevs.get(t.lower(), t) for t in tokens]

    message = " ".join(tokens)

    # Capitalization
    if style["capitalization"]:
        message = message.capitalize()

    # Punctuation
    if random.random() < style["punctuation_rate"]:
        message += random.choice([".", "!", "?"])

    # Emojis
    if random.random() < style["emoji_rate"]:
        message += " " + random.choice(EMOJIS)

    return message


def generate_user(user_id: int, style_name: str) -> dict:
    """Generate a user with consistent style."""
    return {
        "id": str(1000000000 + user_id),
        "username": f"user_{user_id}",
        "discriminator": f"{user_id:04d}",
        "bot": False,
        "style": WRITING_STYLES[style_name],
    }


def generate_synthetic_data(
    num_users: int = 100,
    messages_per_user_range: tuple = (20, 100),
    num_channels: int = 10,
    output_dir: str = "data/raw/synthetic_server",
):
    """Generate synthetic Discord data."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic Discord data...")
    print(f"  Users: {num_users}")
    print(f"  Channels: {num_channels}")

    # Create users with different styles
    users = []
    style_names = list(WRITING_STYLES.keys())
    for i in range(num_users):
        style = random.choice(style_names)
        users.append(generate_user(i, style))

    # Generate messages per channel
    base_timestamp = datetime.now() - timedelta(days=30)

    for channel_id in range(num_channels):
        messages = []
        current_time = base_timestamp

        print(f"  Generating channel {channel_id}...")

        # Each user posts some messages to this channel
        for user in users:
            num_messages = random.randint(*messages_per_user_range)

            for _ in range(num_messages):
                base_msg = random.choice(BASE_MESSAGES)
                content = generate_message(user["style"], base_msg)

                # Add channel-specific context sometimes
                if random.random() < 0.2:
                    content = f"in channel {channel_id}: {content}"

                message = {
                    "id": str(random.randint(100000000000, 999999999999)),
                    "content": content,
                    "timestamp": current_time.isoformat() + "Z",
                    "channel_id": str(8000000000 + channel_id),
                    "author": {
                        "id": user["id"],
                        "username": user["username"],
                        "discriminator": user["discriminator"],
                        "bot": user["bot"],
                    },
                }

                messages.append(message)
                current_time += timedelta(seconds=random.randint(10, 600))

        # Shuffle messages (they're not posted in order)
        random.shuffle(messages)

        # Save channel
        channel_file = output_path / f"channel_{channel_id}.json"
        with open(channel_file, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)

        print(f"    Saved {len(messages)} messages to {channel_file}")

    print(f"\nSynthetic data generated in {output_path}")
    print(f"Total files: {num_channels}")

    # Calculate approximate size
    total_messages = num_users * sum(messages_per_user_range) // 2 * num_channels
    print(f"Approximate messages: {total_messages}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic Discord data")
    parser.add_argument("--users", type=int, default=200, help="Number of users")
    parser.add_argument(
        "--min-messages",
        type=int,
        default=20,
        help="Minimum messages per user per channel",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=100,
        help="Maximum messages per user per channel",
    )
    parser.add_argument("--channels", type=int, default=10, help="Number of channels")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/synthetic_server",
        help="Output directory",
    )

    args = parser.parse_args()

    generate_synthetic_data(
        num_users=args.users,
        messages_per_user_range=(args.min_messages, args.max_messages),
        num_channels=args.channels,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
