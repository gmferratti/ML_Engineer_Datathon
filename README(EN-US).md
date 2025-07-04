# ML\_Engineer\_Datathon

[](https://choosealicense.com/licenses/mit/)

This document reflects the project's structure, which mirrors the organization of the modules contained in the *src/* folder.

-----

## Table of Contents

1.  [Objective and Context](https://www.google.com/search?q=%23objective-and-context)
2.  [Project Overview](https://www.google.com/search?q=%23project-overview)
3.  [Execution Flow](https://www.google.com/search?q=%23execution-flow)
4.  [Configuration and Environment](https://www.google.com/search?q=%23configuration-and-environment)
5.  [Packaging and Deployment](https://www.google.com/search?q=%23packaging-and-deployment)
6.  [Project Structure](https://www.google.com/search?q=%23project-structure)
7.  [Endpoints and Monitoring](https://www.google.com/search?q=%23endpoints-and-monitoring)
8.  [Contributing and Notes](https://www.google.com/search?q=%23contributing-and-notes)
9.  [References](https://www.google.com/search?q=%23references)

-----

## Objective and Context

To develop a personalized recommendation system focused on predicting the next news article a user will read based on their consumption history on **G1**.

> **Note for context:** G1 is a major Brazilian news portal, part of Grupo Globo, the largest media conglomerate in Latin America. Its content is widely consumed across Brazil.

The system is designed to handle both users with an established history and those in a *cold start* situation.

**Team Members:**

  - Antonio Eduardo de Oliveira Lima
  - Gustavo Mendonça Ferratti
  - Luiz Claudio Santana Barbosa
  - Mauricio de Araujo Pintor
  - Rodolfo Olivieri

-----

## Project Overview

The ML\_Engineer\_Datathon is composed of several integrated modules that form a complete recommendation system. Each module has specific documentation, but here is a summary of the main components:

  - **Feature Engineering:**
    Processing of raw news and user data, feature extraction and transformation, and calculation of the engagement score (TARGET).

  - **Training and Ranking:**
    Utilizes **LightGBMRanker** with the *lambdarank* objective to train the model and generate the item ranking, optimizing metrics like NDCG.

  - **Prediction API:**
    Implemented with FastAPI, this API processes inputs, handles *cold start* cases, and returns sorted recommendations with relevant metadata.

  - **Evaluation:**
    A pipeline that uses the **NDCG@10** metric to measure the quality of the ranking generated by the model.

-----

## Execution Flow

1.  **Data Preprocessing:**

      - **News:** Consolidation, filtering, and extraction of information (location, topics, date/time).
      - **Users:** Processing user histories, extracting temporal features, and identifying *cold start* users.
      - **Integration:** Combining data and calculating the TARGET score based on clicks, time on page, scroll depth, recency, and other variables.

    The formula used is:

    ```
    baseScore = numberOfClicksHistory
              + 1.5 * (timeOnPageHistory / 1000)
              + scrollPercentageHistory
              - (minutesSinceLastVisit / 60)
    ```

    Then:

    ```
    rawScore = baseScore * (historySize / 130) * (1 / (1 + (timeGapDays / 50)))
    ```

    Negative values are adjusted, and transformations such as `log1p` and Min-Max Scaling are applied.

2.  **Training and Rank Generation:**

      - The **LightGBMRanker** model is trained to optimize item ordering based on NDCG.
      - During prediction, features are combined to define the recommendation order.

3.  **Prediction API:**

      - The API processes requests, handling inputs and generating different responses for *cold start* cases versus users with an established consumption history.

4.  **Evaluation:**

      - A pipeline calculates **NDCG@10** to measure the effectiveness of the generated ranking, allowing for continuous adjustments and improvements.

-----

## Configuration and Environment

### Main Commands

All commands for executing the modules are defined in the **Makefile**. These include:

  - `make evaluate`
  - `make predict`
  - `make train`
  - Among others.

### Environment Variables

Create a `.env` file in the project root containing at least:

```
ENV="dev"
```

Possible values for `ENV` are `"dev"`, `"staging"`, or `"prod"`.

### Credentials and Servers

  - **AWS:** Configure the necessary credentials in the `.env` file for resource access.
  - **MLflow:** Start the MLflow server (e.g., with `mlflow ui`) using the appropriate URI.
  - **API:** Run the API as specified in the Makefile (e.g., via `uv`).

-----

## Packaging and Deployment

After testing and validation, the system is packaged with Docker and undergoes a rigorous deployment process:

  - **Docker:**

      - **Dockerfile:** Defines the production-optimized container.
      - **docker-compose.yml:** Orchestrates services for the local environment.

  - **Deployment:**
    Additional documentation on deployment is available in `DEPLOY_AWS.md`, which details the process for AWS using ECS Fargate.

-----

## Project Structure

The project is organized as follows:

```
.
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Service orchestration
├── Makefile                # Main project commands
├── README.md               # Main documentation
├── LICENSE                 # Project license
├── pyproject.toml          # Project dependencies
├── requirements.txt        # Project requirements
├── deploy-to-aws.sh        # Script for AWS deployment
├── run-local.sh            # Startup script
├── mlflow.db               # MLflow database
├── mlartifacts/            # MLflow artifacts
├── mlruns/                 # MLflow execution logs
├── uv.lock                 # Requirements lock file (UV)
├── docs/                   # Specific project documentation
├── notebooks/              # Analysis and experiment notebooks
├── tests/                  # Project tests
├── data/                   # Raw or processed data
├── configs/                # Environment configurations
└── src/                    # Source code

```

## Endpoints and Monitoring

### API Endpoints

  - **`GET /health`**: Checks the API's health.

  - **`GET /info`**: Returns information about the model and environment.

  - **`POST /predict`**: Generates recommendations for a user.

    **Example Request:**

    ```json
    {
      "userId": "4b3c2c5c0edaf59137e164ef6f7d88f94d66d0890d56020de1ca6afd55b4f297",
      "max_results": 5,
      "minScore": 0.3
    }
    ```

### Performance Monitoring

The API response includes detailed metrics, such as:

```json
    {
      "processing_time_ms": 123.45,
      "timing_details": {
        "dependencies": 0.01,
        "prediction": 0.12,
        "formatting": 0.01,
        "total_ms": 123.45
      }
    }
```

This information helps in identifying and resolving bottlenecks.

-----

### Contributing

To contribute to the project:

1.  Clone the repository.
2.  Install dependencies with `uv pip install -e .`.
3.  Create a new branch for your changes.
4.  Submit a Pull Request with a clear description of the changes.

-----

### Notes

  - The `run-local.sh` script must be executed from the project root.
  - Ensure consistency in environment settings and documentation across all modules.

-----

## References

  - [LightGBM Documentation](https://lightgbm.readthedocs.io/)
  - [Lambdarank Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)
  - [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
  - [Pandas Documentation](https://pandas.pydata.org/docs/)
  - [FastAPI Documentation](https://fastapi.tiangolo.com/)
