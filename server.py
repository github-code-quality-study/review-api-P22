import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

# Download necessary NLTK resources
nltk.download('vader_lexicon', quiet=True)  # Download VADER sentiment analysis model
nltk.download('punkt', quiet=True)  # Download tokenizer models
nltk.download('averaged_perceptron_tagger', quiet=True)  # Download POS tagger
nltk.download('stopwords', quiet=True)  # Download stop words list

# Initialize sentiment analyzer and stop words list
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Load reviews from CSV into memory
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

# Define a list of allowed locations for validation
ALLOWED_LOCATIONS = [
    'Albuquerque, New Mexico', 'Carlsbad, California', 'Chula Vista, California', 
    'Colorado Springs, Colorado', 'Denver, Colorado', 'El Cajon, California', 
    'El Paso, Texas', 'Escondido, California', 'Fresno, California', 
    'La Mesa, California', 'Las Vegas, Nevada', 'Los Angeles, California', 
    'Oceanside, California', 'Phoenix, Arizona', 'Sacramento, California', 
    'Salt Lake City, Utah', 'San Diego, California', 'Tucson, Arizona'
]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # Placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        """
        Analyzes the sentiment of the review body using VADER.
        Returns sentiment scores including negative, neutral, positive, and compound.
        """
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        Handles HTTP requests. Supports GET for retrieving reviews and POST for adding reviews.
        """
        # Handle GET request
        if environ["REQUEST_METHOD"] == "GET":
            # Parse query parameters from the URL
            query_params = parse_qs(environ['QUERY_STRING'])
            location = query_params.get('location', [None])[0]
            start_date = query_params.get('start_date', [None])[0]
            end_date = query_params.get('end_date', [None])[0]

            # Initialize filtered reviews with all reviews
            filtered_reviews = reviews

            if location:
                # Check if the location is in the allowed list
                if location not in ALLOWED_LOCATIONS:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Invalid location"}).encode("utf-8")]
                # Filter reviews by location
                filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]

            if start_date:
                # Convert start_date to datetime and filter reviews by start date
                start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= start_date_dt]

            if end_date:
                # Convert end_date to datetime and filter reviews by end date
                end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date_dt]

            # Analyze sentiment for each filtered review
            for review in filtered_reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

            # Sort reviews by sentiment compound score in descending order
            filtered_reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

            # Prepare and send response body as JSON
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            start_response("200 OK", [("Content-Type", "application/json"), ("Content-Length", str(len(response_body)))])
            return [response_body]

        # Handle POST request
        if environ["REQUEST_METHOD"] == "POST":
            try:
                # Read and parse the request body
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(request_body_size).decode('utf-8')
                review_data = parse_qs(request_body)
                review_body = review_data.get('ReviewBody', [None])[0]
                location = review_data.get('Location', [None])[0]

                # Validate input data
                if not review_body or not location:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "ReviewBody and Location are required fields"}).encode("utf-8")]

                # Check if the provided location is in the allowed list
                if location not in ALLOWED_LOCATIONS:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Invalid location"}).encode("utf-8")]

                # Generate a unique review ID and timestamp
                review_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Create and add the new review to the reviews list
                new_review = {
                    "ReviewId": review_id,
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": timestamp
                }
                reviews.append(new_review)

                # Prepare and send response body as JSON
                response_body = json.dumps(new_review, indent=2).encode("utf-8")
                start_response("201 Created", [("Content-Type", "application/json"), ("Content-Length", str(len(response_body)))])
                return [response_body]
            except Exception as e:
                # Handle unexpected errors
                start_response("500 Internal Server Error", [("Content-Type", "application/json")])
                return [json.dumps({"error": str(e)}).encode("utf-8")]

if __name__ == "__main__":
    # Start the server on the specified port
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
