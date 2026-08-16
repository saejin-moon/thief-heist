import os

f = "src/train_coop.py"
if os.path.exists(f):
    with open(f, "r") as file:
        content = file.read()
    # Fix the commented out list comprehension
    content = content.replace(
        "[MappoNetwork(state_dim)  # noqa: F821 for _ in range(num_experts)]",
        "[MappoNetwork(state_dim) for _ in range(num_experts)]  # noqa: F821",
    )
    with open(f, "w") as file:
        file.write(content)

print("Fixed syntax error.")
