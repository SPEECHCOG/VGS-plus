import json

coco_root="/worktmp2/hxkhkh/current/FaST/data/coco"
orig_train_json_fn = f"{coco_root}/SpokenCOCO/SpokenCOCO_train.json"
orig_val_json_fn = f"{coco_root}/SpokenCOCO/SpokenCOCO_val.json"
orig_test_json_fn = f"{coco_root}/SpokenCOCO/SpokenCOCO_test_karpathy.json"

unrolled_train_json_fn = f"{coco_root}/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy.json"
unrolled_val_json_fn = f"{coco_root}/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json"
unrolled_test_json_fn = f"{coco_root}/SpokenCOCO/SpokenCOCO_test_unrolled_karpathy.json"


def unroll(orign_json_fn, unrolled_json_fn):
    with open(orign_json_fn) as f:
        orig_data = json.load(f)
    unrolled_data = {"data":[]}
    print(f"unroll {orign_json_fn} and store the unrolled file at {unrolled_json_fn}")
    for item in orig_data['data']:
        for caption in item['captions']:
            unrolled_data["data"].append({"image": item['image'], "caption": caption})
    length = len(unrolled_data["data"])
    print(f"get a total {length} items")
    with open(unrolled_json_fn, "w") as f:
        json.dump(unrolled_data, f)

unroll(orig_val_json_fn, unrolled_val_json_fn)
#unroll(orig_test_json_fn, unrolled_test_json_fn)
#unroll(orig_train_json_fn, unrolled_train_json_fn)


# Khazar: 
# issue 1: There is no Karapathysplit in Kaggle link
# issue 2: I see no difference between the original file and unrolled one (test below)
# issue 3: There is no original test file

# import json
# coco_root="/worktmp2/hxkhkh/current/FaST/data/coco"
# orign_json_fn = f"{coco_root}/SpokenCOCO/SpokenCOCO_val.json"
# with open(orign_json_fn) as f:
#     orig_data = json.load(f)
    
# unrolled_json_fn = f"{coco_root}/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json"
# with open(orign_json_fn) as f:
#     unrolled_data = json.load(f)