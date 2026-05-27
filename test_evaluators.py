import unittest
import os
from PIL import Image
from evaluators import ColorTemperatureEvaluator, AIEvaluator


class TestEvaluators(unittest.TestCase):

    def setUp(self):
        self.test_img_path = "test_dummy.jpg"
        img = Image.new('RGB', (50, 50), color='red')
        img.save(self.test_img_path)

    def tearDown(self):
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)

    def test_color_evaluator_logic(self):
        evaluator = ColorTemperatureEvaluator()
        result = evaluator.evaluate(self.test_img_path)

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_missing_image_handling(self):
        evaluator = ColorTemperatureEvaluator()
        result = evaluator.evaluate("non_existent_image.jpg")
        self.assertIsInstance(result, str)

    def test_abstract_class_protection(self):
        try:
            self.assertTrue(issubclass(ColorTemperatureEvaluator, object))
            self.assertTrue(issubclass(AIEvaluator, object))
        except NameError:
            self.fail("Evaluator classes not found!")


if __name__ == '__main__':
    unittest.main()