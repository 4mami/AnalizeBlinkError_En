import unittest
import PreProcessCirrusDump

class MyTest(unittest.TestCase):
    old_categories = []

    def setUp(self) -> None:
        self.old_categories = list(map(lambda c: c.lower(), [
            'Scooby-Doo', 
            'Fictional characters introduced in 1969', 
            'Media franchises', 
            'Television programs adapted into films', 
            'Television programs adapted into comics'
        ]))
        return super().setUp()

    def test_RoughenCategory(self):
        self.assertEqual(sorted(PreProcessCirrusDump.RoughenCategory(self.old_categories)), [
            '1969', 
            'adapted', 
            'characters', 
            'comics', 
            'fictional', 
            'films', 
            'franchises', 
            'into', 
            'introduced', 
            'media', 
            'programs', 
            'scooby-doo', 
            'television'
        ])

if __name__ == '__main__':
    unittest.main()