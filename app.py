from flask import Flask, request, render_template
import numpy as np
import cv2
import os
from tensorflow import keras
import base64
import requests
import json
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
app = Flask(__name__)

# Load the saved MobileNet model
loaded_model = keras.models.load_model('./models/mobilenet-bs32-e20_old.h5')

activities = [
    ("Lễ hội Lồng Tồng","Trò chơi ném còn"), 
    ("Lễ hội Tháp Bà Ponagar","Biểu diễn múa Chăm"),
    ("Lễ hội Chùa Hương","Tham quan động Hương Tích"), 
    ("Hội Lim","Hát Quan Họ"), 
    ("Hội Lim","Trò chơi đánh đu"), 
    ("Lễ hội Cổ Loa","Đánh cờ người"), 
    ("Lễ hội Cổ Loa","Trò chơi bắn nỏ"), 
    ("Lễ hội Ok Om Bok","Đua ghe ngo"), 
    ("Lễ hội đua voi Tây Nguyên","Hoạt động đua voi"), 
    ("Lễ hội đua bò Bảy Núi","Hoạt động đua bò"), 
    ("Lễ hội Chùa Hương","Đi thuyền trên sông"), 
    ("Lễ hội Lồng Tồng","Trò chơi đi cà kheo"), 
    ("Lễ hội Ok Om Bok","Hoạt động thả đèn nước"), 
    ("Lễ hội Ok Om Bok","Hoạt động thả đèn trời"), 
]
activities1 = [
    "Trò chơi ném còn", "Biểu diễn múa Chăm", "Tham quan động Hương Tích", 
    "Hát Quan Họ", "Trò chơi đánh đu", "Đánh cờ người", "Trò chơi bắn nỏ",
    "Đua ghe ngo", "Hoạt động đua voi", "Hoạt động đua bò", 
    "Đi thuyền trên sông", "Trò chơi đi cà kheo", 
    "Hoạt động thả đèn nước", "Hoạt động thả đèn trời"
]
conflict_pairs = [
    ("Trò chơi ném còn", "Đi thuyền trên sông"),
    ("Trò chơi ném còn", "Đua ghe ngo"),
    ("Trò chơi ném còn", "Hoạt động thả đèn nước"),
    ("Hát Quan Họ", "Đi thuyền trên sông"),
    ("Hát Quan Họ", "Đua ghe ngo"),
    ("Hát Quan Họ", "Hoạt động thả đèn nước"),
    ("Trò chơi đánh đu", "Đi thuyền trên sông"),
    ("Trò chơi đánh đu", "Đua ghe ngo"),
    ("Trò chơi đánh đu", "Hoạt động thả đèn nước"),
    ("Đánh cờ người", "Đi thuyền trên sông"),
    ("Đánh cờ người", "Đua ghe ngo"),
    ("Đánh cờ người", "Hoạt động thả đèn nước"),
    ("Trò chơi đi cà kheo", "Đi thuyền trên sông"),
    ("Trò chơi đi cà kheo", "Đua ghe ngo"),
    ("Trò chơi đi cà kheo", "Hoạt động thả đèn nước"),
    ("Trò chơi bắn nỏ", "Đi thuyền trên sông"),
    ("Trò chơi bắn nỏ", "Đua ghe ngo"),
    ("Trò chơi bắn nỏ", "Hoạt động thả đèn nước"),
    ("Hoạt động đua voi", "Đi thuyền trên sông"),
    ("Hoạt động đua voi", "Đua ghe ngo"),
    ("Hoạt động đua voi", "Hoạt động thả đèn nước"),
    ("Hoạt động đua bò", "Đi thuyền trên sông"),
    ("Hoạt động đua bò", "Đua ghe ngo"),
    ("Hoạt động đua bò", "Hoạt động thả đèn nước"),
    ("Hoạt động thả đèn trời", "Đi thuyền trên sông"),
    ("Hoạt động thả đèn trời", "Đua ghe ngo"),
    ("Biểu diễn múa Chăm", "Đi thuyền trên sông"),
    ("Biểu diễn múa Chăm", "Đua ghe ngo"),
    ("Biểu diễn múa Chăm", "Hoạt động thả đèn nước"),
    ("Tham quan động Hương Tích", "Đua ghe ngo"),
    ("Tham quan động Hương Tích", "Thả đèn nước"),

]

HEIGHT = 128
WIDTH = 128
def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=20.0))
    video_manager.set_downscale_factor()
    
    video_manager.start()
    scene_manager.detect_scenes(video_manager)
    scenes = scene_manager.get_scene_list()
    video_manager.release()
    
    return [(start.get_seconds(), end.get_seconds()) for start, end in scenes]

def process_video(video_file):
    # Detect scenes
    scenes = detect_scenes(video_file)
    cap = cv2.VideoCapture(video_file)
    
    results_subtitle = []
    festival_counts = {activity[0]: 0 for activity in activities}  # Initialize festival counts

    for start_sec, end_sec in scenes:
        cap.set(cv2.CAP_PROP_POS_MSEC, (start_sec + end_sec) / 2 * 1000)  # Get frame in the middle of the scene
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Use predict_image to get the top activity
        predictions = predict_image(frame)
        top_index = np.argmax(predictions)
        top_activity = activities[top_index][1]
        top_festivals = activities[top_index][0]  # Get the festival name
        results_subtitle.append((start_sec, end_sec, top_activity, predictions[top_index]))
        festival_counts[top_festivals] += 1  # Count the detected activity

    cap.release()
    return results_subtitle, festival_counts


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def resize_image(image, target_size):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    if w > h:
        new_w = target_size
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = target_size
        new_w = int(new_h * aspect_ratio)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    color = [0, 0, 0]
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def predict_image(image):
    image = resize_image(image, max(WIDTH, HEIGHT))
    image = np.array(image, dtype="float") / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = loaded_model.predict(image)
    return prediction[0]

def detect_conflict(top_activities, conflict_pairs):
    conflicts = []
    for i in range(len(top_activities)):
        for j in range(i + 1, len(top_activities)):
            if (top_activities[i], top_activities[j]) in conflict_pairs:
                conflicts.append((i, j))
    return conflicts

def determine_most_likely_festival(festival_counts, threshold=2):
    sorted_festivals = sorted(festival_counts.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_festivals) > 1 and (sorted_festivals[0][1] - sorted_festivals[1][1]) >= threshold:
        return sorted_festivals[0][0]
    return None

# Load festival data from JSON file
with open('./static/json/festivals.json', 'r', encoding='utf-8') as f:
    festival_data = json.load(f)
def get_festival_info(festival_name):
    # Retrieve festival information from the loaded JSON data
    return festival_data.get(festival_name, "No information available.")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('images')
        file_count = len(files)
        if file_count < 7 or file_count > 10:
            return render_template('index.html', error="Để nhận diện chính xác hơn vui lòng tải lên từ 7 đến 10 ảnh.")
        
        results = []
        festival_counts = {}
        for activity in activities:
            festival_counts[activity[0]] = 0
        top_activities = []  # Store top activity of each image

        for file in files:
            if file:
                # Read image
                image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
                
                # Convert original image to displayable format
                _, buffer = cv2.imencode('.jpg', image)
                img_str = base64.b64encode(buffer).decode('utf-8')
                img_data = f"data:image/jpeg;base64,{img_str}"

                # Predict
                predictions = predict_image(image)
                top_2_indices = np.argsort(predictions)[-2:][::-1]
                top_2_labels = []
                for idx in top_2_indices:
                    top_2_labels.append((activities[idx], predictions[idx]))

                # Store the top activity for conflict checking
                top_activities.append(top_2_labels[0][0][1])

                # # Count festivals
                # for label, _ in top_2_labels:
                #     festival_counts[label[0]] += 1
                # Count festivals
                top_1_label, top_1_prob = top_2_labels[0]  # Lấy top 1
                festival_counts[top_1_label[0]] += 1  # Đếm top 1

                # Chỉ đếm top 2 nếu tỉ lệ lớn hơn 0.1
                if top_2_labels[1][1] > 0.1:
                    festival_counts[top_2_labels[1][0][0]] += 1

                results.append((img_data, top_2_labels))

        # Check for conflicts between top activities of different images
        conflicts = detect_conflict(top_activities, conflict_pairs)

        if conflicts:
            conflict_images = [(results[i][0], results[j][0]) for i, j in conflicts]
            return render_template(
                'index.html', 
                results=results, 
                conflict_images=conflict_images, 
                message="Các hình ảnh bạn cung cấp xuất hiện sự không đồng nhất, đảm bảo bạn đã tải lên những hình ảnh liên quan đến cùng một lễ hội"
            )
        else:
            most_likely_festival = determine_most_likely_festival(festival_counts, threshold=2)
            if most_likely_festival:
                festival_info = get_festival_info(most_likely_festival)
                return render_template(
                    'index.html', 
                    results=results, 
                    message=f"{most_likely_festival}", festival_info=festival_info,
                    predicted = True
                )
            else:
                return render_template(
                    'index.html', 
                    results=results, 
                    message="Không thể kết luận các hình ảnh bạn cung cấp liên quan đến lễ hội nào vì kết quả dự đoán chưa chiếm số đông bởi bên nào"
                )

    return render_template('index.html')

@app.route('/video', methods=['POST'])
def process_video_upload():
    if 'video' not in request.files:
        return render_template('index.html', error="Vui lòng tải lên một video.")

    video_file = request.files['video']
    if video_file:
        # Ensure the uploads directory exists
        uploads_dir = './static/uploads'
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)

        video_path = os.path.join(uploads_dir, video_file.filename)
        video_file.save(video_path)

        # Process the video and generate subtitles
        results_subtitle, festival_counts = process_video(video_path)
        vtt_file = os.path.splitext(video_path)[0] + '.vtt'
        generate_vtt_file(results_subtitle, vtt_file)

        # Determine the most likely festival
        most_likely_festival = determine_most_likely_festival(festival_counts, threshold=2)
        festival_info = None
        if most_likely_festival:
            festival_info = get_festival_info(most_likely_festival)
            return render_template(
                'index.html',  
                video_url=video_file.filename,  # Just the filename
                vtt_url=os.path.basename(vtt_file),  # Just the filename
                video_uploaded=True,
                message=f"{most_likely_festival}", festival_info=festival_info,
                predicted = True
            )
        
        else:
            return render_template(
                'index.html', 
                message="Không thể kết luận các Video bạn cung cấp liên quan đến lễ hội nào vì kết quả dự đoán chưa chiếm số đông bởi bên nào"
            )

    return render_template('index.html', error="Đã xảy ra lỗi khi tải lên video.")

def generate_vtt_file(results, vtt_file):
    with open(vtt_file, 'w') as f:
        f.write("WEBVTT\n\n")  # Write the WebVTT header
        for i, (start_sec, end_sec, activity, confidence) in enumerate(results, start=1):
            start_time = format_time(start_sec)
            end_time = format_time(end_sec)
            subtitle_text = f"HĐ: {activity} (Độ tin cậy: {confidence:.2f})"
            f.write(f"{start_time} --> {end_time}\n{subtitle_text}\n\n")

if __name__ == '__main__':
    app.run(debug=True)
