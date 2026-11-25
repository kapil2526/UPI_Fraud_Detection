# Dockerfile - run the Streamlit app in a container
# Build (from the main project folder):  docker build -f deployment/Dockerfile -t upi-fraud .
# Run:                                   docker run -p 8501:8501 upi-fraud

FROM python:3.12-slim

WORKDIR /project

# Install the libraries first (this step is cached by Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project (the model file ml/artifacts/best_model.pkl must exist)
COPY . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]