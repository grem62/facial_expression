import os
import matplotlib.pyplot as plt

# Function to count the number of images in each emotion folder
def count_images_in_emotion_folders(folder_path):
    # List of emotions corresponding to folder names
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    # List of valid image file extensions
    image_extensions = ('.jpg')
    # Dictionary to store the count of images for each emotion
    emotion_counts = {}

    # Loop through each emotion to count images
    for emotion in emotions:
        # Construct the path to the emotion folder
        emotion_folder = os.path.join(folder_path, emotion)
        
        # Check if the emotion folder exists
        if not os.path.exists(emotion_folder):
            print(f"The folder '{emotion}' does not exist in {folder_path}.")
            emotion_counts[emotion] = 0
            continue
        
        # List all files in the emotion folder
        files = os.listdir(emotion_folder)
        # Filter out only image files
        image_files = [file for file in files if file.lower().endswith(image_extensions)]
        # Store the count of image files in the dictionary
        emotion_counts[emotion] = len(image_files)

    return emotion_counts

# Path to the main folder containing emotion subfolders
folder_path = "/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/raw/CK+"
# Get the count of images for each emotion
image_counts = count_images_in_emotion_folders(folder_path)
# Print the count of images for each emotion
for emotion, count in image_counts.items():
    print(f"Number of images in the '{emotion}' folder: {count}")

# Prepare data for the histogram
emotions = list(image_counts.keys())
counts = list(image_counts.values())

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.bar(emotions, counts, color='skyblue')
plt.xlabel('Emotions')
plt.ylabel('Number of Images')
plt.title('Number of Images per Emotion')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the histogram to a file
output_path = "/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/reports/figures_data/histogram.png"
plt.savefig(output_path)
plt.show()
plt.close()

print(f"Histogram saved to {output_path}")
