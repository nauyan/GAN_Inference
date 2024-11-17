# GAN-Inference

A data transformation service that processes large-scale datasets using GAN-based inference.

## Installation & Running

### Without Docker
#### Install dependencies:
```bash
pip install -r requirements.txt
```
#### Run the application:
```bash
python app.py
```

### With Docker
```bash
docker compose up -d
```

## API Documentation
Transform Data Endpoint
API Endpoint: `/transforming_data`
Method: `POST`
Description: Processes a Parquet shard file to perform data transformation and inverse transformation.
Request Body
```json
{
"file_path": "string"
}
```
### Example Usage
```bash
curl -X POST "http://localhost:8000/transforming_data" 
-H "Content-Type: application/json" 
-d '{"file_path": "/app/data/Amount_Shard_2.parquet"}'
```
### Example Response
```json
{
"total_time_taken": "3.56 seconds",
"transformed_file_path": "/app/output/transform_f29f4888-4041-4ca2-bdb8-6dc296faec0d",
"inverse_transformed_file_path": "/app/output/inverse_transform_f29f4888-4041-4ca2-bdb8-6dc296faec0d"
}
```

## System Overview

This system processes data in several steps to efficiently handle and transform large datasets. The workflow is divided into the following stages:

### 1. Initial Data Generation

- **Script Used**: `create_one_billion_csv.py`
- **Process**:
  - Generates a large CSV file containing one billion records.
  - The CSV file is then split into smaller **Parquet shards** for efficient processing.
  - These Parquet shards are stored in the `data` folder for further processing.

### 2. Model Processing

- **Process**:
  - Reads Parquet shards from the `data` folder.
  - The system fits a model on all shards, creating **model** and **metadata** files (`model.pkl` and associated metadata) for transforming and inverse transforming the data.
  - The model's state and metadata are saved for both the transformation and inverse transformation processes.

### 3. Data Transformation

- **Process**:
  - The system uses the saved metadata to transform the values for each shard.
  - Transformed data is stored as new **Parquet files**.
  - The file paths of the transformed data are returned after the transformation is completed.

### 4. Inverse Transformation

- **Process**:
  - The system loads the transformed Parquet files.
  - It applies the inverse transformation using the same model.
  - The resulting inverse-transformed data is saved in new Parquet files.
  - The file paths of the inverse-transformed data are returned.

## Performance Optimizations

The system includes several optimizations to handle large datasets more efficiently:

- **File Sharding**: The large CSV file is split into smaller Parquet shards for easier processing.
- **Warm Start**: The model retains its context by using a warm start, which reduces computation time for each shard.
- **Model and Metadata Caching**: By saving the model state and metadata, the system avoids recalculating the transformations and inverse transformations from scratch.

## Data Generation
To generate the initial large dataset and its shards, run:
```bash
python create_one_billion_csv.py
```

## Trade-Offs Between Scalability, Speed, and Correctness
To ensure scalability, we shard the dataset. For speed, we train the model on 10k-100k samples per shard, which reduces accuracy slightly but improves performance. Correctness may be affected by smaller sample sizes, but this trade-off allows us to balance all three factors efficiently.