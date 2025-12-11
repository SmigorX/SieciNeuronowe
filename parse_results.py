#!/usr/bin/env python3
import re
import sys

# Input file
FILE = "./results"

# Regex patterns
block_start = re.compile(r"^=+")
learning_params = re.compile(r"^Learning Parameters:")
test_acc_pattern = re.compile(r"test_accuracy:\s*([0-9.]+)")

best_acc = -1
best_block = []
collecting = False
current_block = []

with open(FILE, "r") as f:
    for line in f:
        # start of a new block (e.g., "========")
        if block_start.match(line):
            current_block = [line]
            collecting = False
            continue

        # block content before learning parameters
        if not collecting:
            current_block.append(line)

        # detect start of Learning Parameters
        if learning_params.match(line):
            collecting = True
            continue

        # if collecting the block after Learning Parameters
        if collecting:
            current_block.append(line)

            # test_accuracy line detection
            match = test_acc_pattern.search(line)
            if match:
                acc = float(match.group(1))
                if acc > best_acc:
                    best_acc = acc
                    best_block = current_block[:]

                # block ends after test_accuracy
                collecting = False

# Output result
if best_block:
    print("".join(best_block))
else:
    print("No test_accuracy found.")
