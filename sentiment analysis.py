from flask import Flask, render_template, request
import cv2
from fer import FER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
from collections import Counter

emotion_detector = FER()
webcam = cv2.VideoCapture(0)


analyzer = SentimentIntensityAnalyzer()


def analyze_feedback(feedback):
    sentiment_scores = analyzer.polarity_scores(feedback)
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.01:
        return "Negative"
    else:
        return "Neutral"


def analyze_facial_emotion():
    detected_emotions = []
    start_time = time.time()
    duration = 10

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to capture video frame.")
            break


        emotions = emotion_detector.detect_emotions(frame)
        if emotions:
            emotion_dict = emotions[0]['emotions']
            highest_emotion = max(emotion_dict, key=emotion_dict.get)
            detected_emotions.append(highest_emotion)

            cv2.putText(frame, f"Emotion: {highest_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        cv2.imshow('Facial Emotion Recognition', frame)


        if time.time() - start_time > duration:
            break

        if cv2.waitKey(10) & 0xFF == ord('q') or cv2.waitKey(10) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

    if detected_emotions:
        most_common_emotion = Counter(detected_emotions).most_common(1)[0][0]
        return most_common_emotion
    else:
        return "No emotion detected."


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    feedback = request.form['feedback']
    sentiment = analyze_feedback(feedback)
    return render_template('result.html', result=f"The feedback is classified as: {sentiment}")


@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    emotion = analyze_facial_emotion()
    return render_template('result.html', result=f"The most frequently detected emotion: {emotion}")


if __name__ == "__main__":
    app.run(debug=True)
