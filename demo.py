from dotenv import load_dotenv
from src.pipline.training_pipeline import TrainPipeline

load_dotenv()

train_pipeline = TrainPipeline()
train_pipeline.run_pipeline()
