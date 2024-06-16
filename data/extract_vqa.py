import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import os

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
parser = argparse.ArgumentParser(description="Extracting bottom-up features")
parser.add_argument("--input", type=str, required=True, help="path to bottom-up features")
parser.add_argument("--output", type=str, required=True, help="path to saving the extracted features")
args = parser.parse_args()

if __name__ == '__main__':

    # Verify we can read a tsv
    in_data = {}
    with open(args.input, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            img_id = int(item['image_id'])
            cur_data = np.frombuffer(base64.decodestring(item['features']),
                  dtype=np.float32).reshape((int(item['num_boxes']),-1))
            np.save(os.path.join(args.output,str(img_id)),cur_data)
