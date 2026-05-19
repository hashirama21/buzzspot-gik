#!/usr/bin/env python3

import argparse
import json
import os
from zipfile import ZipFile
from pycocotools.coco import COCO


class ValidationException(Exception):
  pass


def run():
  parser = argparse.ArgumentParser(
      description=
      "Validate a submission zip file needed to evaluate on Codabench competition of BuzzSpot.",
      formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument(
      "--zipfile",
      type=str,
      required=True,
      help='zip file that should be validated.',
  )

  parser.add_argument('--test_file',
                      type=str,
                      required=True,
                      help='Path to test.json file needed to check if the prediction file contains predictions for the key-frame test images.')

  FLAGS, _ = parser.parse_known_args()
  FLAGS = parser.parse_args()

  checkmark = "\u2713"


  try:

    print('Validating zip archive "{}".\n'.format(FLAGS.zipfile))


    print("  1. Checking filetype............................................\t", end="", flush=True)
    if not FLAGS.zipfile.endswith('.zip'):
      raise ValidationException('Competition submission must end with ".zip"')
    print(checkmark)


    with ZipFile(FLAGS.zipfile) as zipfile:
      print('  2. Checking files ')
      prediction_files = {info.filename: info for info in zipfile.infolist() if not info.filename.endswith("/")}
      annotation_file = FLAGS.test_file

      print("\t Check if prediction file exists..........................\t", end="", flush=True)


      expected_file = "predictions.json"
      if expected_file not in prediction_files:
          raise ValidationException(f'Missing prediction file {expected_file} in {FLAGS.zipfile}!')
      print(checkmark)
      print("\t Check if test.json file exists...........................\t", end="", flush=True)
      if not os.path.exists(annotation_file):
          raise ValidationException(f'Missing test file {"test.json"} in {os.path.dirname(annotation_file)}!\n'
                                    'test.json is needed to check if the prediction file contains predictions for the key-frame test images.')
      print(checkmark)


      print("  3. Checking test.json\n\t(needed for consistency check with prediction file)\n")
      try:
        test_set = COCO(annotation_file)
        #Check if has "is_keyframe" field in test.json
        img_info = test_set.loadImgs(test_set.getImgIds()[0])[0]
        if "is_keyframe" not in img_info:
            raise ValidationException(f'"is_keyframe" field missing in test.json annotations!\n'
                                      'This field is needed to check if the prediction file contains predictions for the key-frame test images.\n')
      except Exception as ex:
        raise ValidationException(f'Error loading annotation file {annotation_file}!\n'
                                  f'Error message: {str(ex)}')

      def get_image_ids_of_keyframes(coco):
          image_ids = coco.getImgIds()
          keyframe_image_ids = []
          for image_id in image_ids:
              image_info = coco.loadImgs(image_id)[0]
              if image_info.get("is_keyframe", False):
                  keyframe_image_ids.append(image_id)   
          return set(keyframe_image_ids)
      
      keyframe_image_ids = get_image_ids_of_keyframes(test_set)


      print(f"\t\t\t\t\t\t\t\t\t{checkmark}")
      print("  4. Check if predictions.json follow the eval format ")

      print("\t[")
      print("\t{'image_name': 'buzzspot_00001.png', 'category_id': 1, 'bbox': [120, 80,  60, 70], 'score': 0.91},")
      print("\t{'image_name': 'buzzspot_00001.png', 'category_id': 3, 'bbox': [300, 200, 50, 50], 'score': 0.74},")
      print("\t{'image_name': 'buzzspot_00002.png', 'category_id': 2, 'bbox': [10 , 10 , 80, 90], 'score': 0.88}")
      print( "\t]\n\t(see example above for expected format)\n")

      
      with zipfile.open(prediction_files[expected_file]) as pred_file:

        
        predictions = json.load(pred_file)
        pred_image_ids = set(pred['image_id'] for pred in predictions)
        #pred_image_ids.add(5)
        non_keyframe_image_ids = pred_image_ids - keyframe_image_ids
        if non_keyframe_image_ids:
            raise ValidationException(f'Predictions for non-keyframe images found in {expected_file}!\n'
                                      f'Non-keyframe image ids: {non_keyframe_image_ids}')
        expected_classes = set(range(1, 6)) 
        ann_image_ids = set()
        for ann in predictions:
            for key in ( "image_id", "category_id", "bbox", "score"):
                if key not in ann:
                    raise ValidationException(f"Annotation missing '{key}': {ann}")
            if ann["image_id"] not in keyframe_image_ids:
                raise ValidationException(f"Annotation references non-keyframe image_id: {ann['image_id']}")
            if ann["category_id"] not in expected_classes:
                raise ValidationException(f'Annotation references unknown category_id: {ann['category_id']}\n'
                                          f'Expected category_ids: {expected_classes}')   
            bbox = ann.get("bbox")
            if bbox:
                if len(bbox) != 4:
                    raise ValidationException(f"bbox must have 4 values [x, y, w, h]: {bbox}")
                if bbox[2] <= 0 or bbox[3] <= 0:
                    raise ValidationException(f"bbox w/h must be positive: {bbox}")
            ann_image_ids.add(ann["image_id"])


      print(f"\t Check if all annotations have valid keys..................\t{checkmark}")
      print(f"\t Check if predictions are only on key-frame test images....\t{checkmark}")
      print(f"\t Check for unexpected classes..............................\t{checkmark}")
      print(f"\t Check if bbox format is correct...........................\t{checkmark}")
      print(f"\t Check if bbox values are valid............................\t{checkmark}")


  except ValidationException as ex:
    print("\n\n  " + "\u001b[1;31m>>> Error: " + str(ex) + "\u001b[0m")
    exit(1)

  print("\n\u001b[1;32mEverything ready for submission!\u001b[0m  \U0001F389")


if __name__ == "__main__":
  run()
