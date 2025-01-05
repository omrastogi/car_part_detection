#%%
import autoroot
import autorootcwd
import os
import json
import pandas as pd
# Load annotations
annotations = []
base_dir = "data"  # Replace with your actual path
for root, _, files in os.walk(base_dir):
    for file in files:
        if file == "via_region_data.json":
            with open(os.path.join(root, file), 'r') as f:
                data = json.load(f)
                for key, annotation in data.items():  # Assuming VIA annotations are stored in a dictionary
                    annotation["path"] = os.path.relpath(root, base_dir)
                annotations.extend(data.values())  # Assuming VIA annotations are stored in a dictionary
# %%
annotation_to_class = {
    "Front": [
        "bonnet", "frontbumper", "frontws", "headlightwasher", "indicator",
        "leftheadlamp", "rightheadlamp", "frontbumpergrille", "lowerbumpergrille",
        "licenseplate", "namebadge"
    ],
    "Rear": [
        "rearbumper", "rearws", "fuelcap", "taillamp", "rearbumpercladding",
        "leftbootlamp", "rightbootlamp", "towbarcover", "lefttailgate",
        "righttailgate", "rearbumpermissing", "rearwsmissing"
    ],
    "Front-Right": [
        "rightfender", "rightfrontdoor", "rightfrontdoorglass", "rightorvm",
        "rightfoglamp", "partial_rightfender", "partial_rightfrontdoor",
        "rightfrontbumper"
    ],
    "Front-Left": [
        "leftfender", "leftfrontdoor", "leftfrontdoorglass", "leftorvm",
        "leftfoglamp", "partial_leftfender", "partial_leftfrontdoor",
        "leftfrontbumper"
    ],
    "Rear-Right": [
        "rightqpanel", "rightreardoor", "rightreardoorglass", "rightrearventglass",
        "partial_rightqpanel", "partial_rightreardoor", "rightrearbumper"
    ],
    "Rear-Left": [
        "leftqpanel", "leftreardoor", "leftreardoorglass", "leftrearventglass",
        "partial_leftqpanel", "partial_leftreardoor", "leftrearbumper"
    ],
    "None": [
        "alloywheel", "antenna", "car", "cracked", "dirt", "logo", "reflection",
        "rust", "scratch", "shattered", "sensor", "sunroof", "wiper", "series"
    ]
}

#%%

# Initialize the result list
image_data = []

#########################################################
# THRESHOLD_FACTOR: how many times bigger "None" must be
# than any other label to be chosen as the final label.
#########################################################
THRESHOLD_FACTOR = 2.0  

for annotation in annotations:
    file_name = annotation['filename']
    regions = annotation.get('regions', [])
    
    annotation_list = []
    annotation_areas = []
    class_labels = []
    class_areas = []
    
    # Initialize the label area map with every known class + 'None' to 0
    label_area_map = {label: 0 for label in annotation_to_class.keys()}
    label_area_map['None'] = 0

    # --- Process each region ---
    for region in regions:
        shape_attributes = region['shape_attributes']
        region_attributes = region['region_attributes']

        # Extract the annotation name from 'identity'
        annotation_name = region_attributes.get('identity', 'None')

        # Determine which class label it belongs to (if any)
        label = next(
            (key for key, values in annotation_to_class.items() 
             if annotation_name in values),
            'None'
        )

        # Calculate the area of the annotation
        if shape_attributes.get('name') == 'polygon':
            x = shape_attributes['all_points_x']
            y = shape_attributes['all_points_y']
            area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] 
                                 for i in range(-1, len(x)-1)))
        else:
            # In case of other shapes, set area to 0 or handle differently
            area = 0

        # Append the raw annotation data
        annotation_list.append(annotation_name)
        annotation_areas.append(area)

        # If we haven't seen this class label yet, initialize; otherwise add to it
        if label not in class_labels:
            class_labels.append(label)
            class_areas.append(area)
        else:
            class_areas[class_labels.index(label)] += area

        # Update the total area for this label
        label_area_map[label] += area

    # --- Determine the final label --- 
    if not annotation_list:
        # If there are NO annotations at all, final_label = 'None'
        final_label = 'None'
    else:
        # Separate 'None' area from other labels' areas
        none_area = label_area_map.get('None', 0)
        # Dictionary with everything except 'None'
        label_area_map_no_none = {
            k: v for k, v in label_area_map.items() if k != 'None'
        }

        if not label_area_map_no_none:
            # If the only label is 'None', final_label = 'None'
            final_label = 'None'
        else:
            # Find the label with the largest area among non-'None'
            max_non_none_label = max(label_area_map_no_none, key=label_area_map_no_none.get)
            max_non_none_area = label_area_map_no_none[max_non_none_label]

            # Compare 'None' area vs. largest non-'None' area
            if none_area > THRESHOLD_FACTOR * max_non_none_area:
                final_label = 'None'
            else:
                final_label = max_non_none_label
    
    # Optional: print debug info
    print(f"label_area_map: {label_area_map} -> final_label: {final_label}")

    # --- Append data to the result list ---
    image_data.append({
        "path": annotation["path"],
        'image': file_name,
        'annotations': annotation_list,
        'ann_areas': annotation_areas,
        'class_labels': class_labels,
        'class_areas': class_areas,
        'final_label': final_label
    })

# Convert to DataFrame
df = pd.DataFrame(image_data)
print("\nFinal DataFrame:")
df.head()


#%%
# Add a new column for multi-class labels
def calculate_multi_class_labels(row, threshold=0.1):
    if row["class_areas"] == []:
        return ['None']
    largest_area = max(row["class_areas"])
    multi_class_labels = [
        label for label, area in zip(row["class_labels"], row["class_areas"])
        if area >= threshold * largest_area
    ]
    return multi_class_labels

df["multi_class_labels"] = df.apply(calculate_multi_class_labels, axis=1)
df.head()

# %%
from sklearn.model_selection import train_test_split

# Stratified split into train and temp (val + test)
train_df, temp_df = train_test_split(
    df,
    test_size=0.08,  # Adjust this based on desired val + test size
    random_state=42,
    shuffle=True,
    stratify=df["final_label"]  # Use the column containing class labels
)

# Stratified split of temp into val and test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,  # 50% of temp goes to test, 50% to val
    random_state=42,
    stratify=temp_df["final_label"]
)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
