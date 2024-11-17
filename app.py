import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import time
from src.model import DataTransformer
from src.config import (
    DATA_DIR,
    TRANSFORM_ROW_LIMIT,
    UVICORN_HOST,
    UVICORN_PORT,
    UVICORN_RELOAD,
)
import uuid
import os

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Data Model
class TransformRequest(BaseModel):
    file_path: str


@app.post("/transforming_data")
async def transforming_data(request: TransformRequest):
    """
    Endpoint for transforming and inverse transforming a dataset.

    Args:
        request (TransformRequest): Contains the file path of the CSV shard.

    Returns:
        dict: Contains total time taken, transformed file path, and inverse transformed file path.
    """
    file_path = request.file_path
    logger.info(f"Received request for file: {file_path}")

    # Ensure the file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    start_time = time.time()

    # Initialize transformer
    job_id = uuid.uuid4()
    transformer = DataTransformer(job_id, DATA_DIR)
    logger.info(f"Initialized DataTransformer with job_id: {job_id}")

    # Fit the transformer
    try:
        transformer.fit()
        logger.info("Transformer fit successfully")
    except Exception as e:
        logger.error(f"Error during fitting transformer: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error during fitting transformer: {str(e)}"
        )

    # Read and transform the data
    try:
        train_data = pd.read_parquet(file_path)
        logger.info("File read successfully")
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    # Transform the data
    try:
        transformed_file_path = transformer.transform(
            train_data.values[:TRANSFORM_ROW_LIMIT]
        )
        logger.info(
            f"Data transformed successfully. Transformed file saved at: {transformed_file_path}"
        )
    except Exception as e:
        logger.error(f"Error during transformation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error during transformation: {str(e)}"
        )

    # Inverse transform the data
    try:
        transformed_data = pd.read_parquet(transformed_file_path)
        inverse_transformed_file_path = transformer.inverse_transform(
            transformed_data.values
        )
        logger.info(
            f"Inverse transformation completed. File saved at: {inverse_transformed_file_path}"
        )
    except Exception as e:
        logger.error(f"Error during inverse transformation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error during inverse transformation: {str(e)}"
        )

    total_time = time.time() - start_time
    logger.info(f"Total time taken for processing: {total_time:.2f} seconds")

    return {
        "total_time_taken": f"{total_time:.2f} seconds",
        "transformed_file_path": transformed_file_path,
        "inverse_transformed_file_path": inverse_transformed_file_path,
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting FastAPI server at {UVICORN_HOST}:{UVICORN_PORT}")
    uvicorn.run("app:app", host=UVICORN_HOST, port=UVICORN_PORT, reload=UVICORN_RELOAD)
