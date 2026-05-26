import os
from evaluators import AIEvaluator, ColorTemperatureEvaluator

def main():
    print("=== Starting Polymorphic Evaluation ===\n")

    # Initialize both distinct evaluator objects
    ai_judge = AIEvaluator("hotness_model.pth")
    color_judge = ColorTemperatureEvaluator()

    # The core of polymorphism: grouping different objects under one interface
    judges = [ai_judge, color_judge]

    # Pick a random target image from your validation set
    # Make sure this specific file exists in your dataset/val folder!
    target_image = "C:/Users/andre/Downloads/l104.jpg"

    print(f"Target Image: {target_image}")
    print("-" * 40)

    # Iterate blindly through the list calling the exact same method
    for judge in judges:
        print(f"Evaluator Logic: {judge.name}")
        result = judge.evaluate(target_image)
        print(f"Result: {result}\n")

if __name__ == "__main__":
    main()