# GraphRAG-Architecture-of-a-local-LLM-for-Gestational-Diabetes-Mellitus

## Overview

This project aims to build a Graph-RAG (Retrieval-Augmented Generation) architecture for a local language model focused on Gestational Diabetes Mellitus (GDM). It combines the latest advancements in natural language processing and knowledge graph construction to provide meaningful insights by querying medical research articles related to GDM.

**Citation:**

F. Ruba, A. Nazir, E. Evangelista, S. Bukhari, L. bin Mohd Lofti & R. Sharma. (2025.) Data Repository for GraphRAG-Architecture-of-a-local-LLM-for-Gestational-Diabetes-Mellitus. Available at: https://github.com/zuhealth-wellness/GDMChatbot

---

## Prerequisites

Before running this project, ensure you have the following:

1. **Google Colab Account**: This project is designed to run on Google Colab
2. **Neo4j Database**: You'll need a Neo4j instance (recommended: Neo4j Aura for cloud-based setup)
3. **OpenAI API Key**: Required for entity extraction using GPT-3.5 Turbo

---

## Setup Instructions

### Step 1: Setting Up Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook or upload the project files
3. Upload the project files to your Colab environment:
   - `Data Collection.py`
   - `Entity Extraction and Construction of Knowledge Graph.py`

### Step 2: Setting Up Neo4j

#### Option A: Neo4j Aura (Recommended for Colab)

1. Go to [Neo4j Aura](https://neo4j.com/cloud/aura/) and create a free account
2. Create a new free instance (Free tier available)
3. After creation, you'll receive:
   - **Neo4j URL**: Format: `neo4j+s://xxxxx.databases.neo4j.io`
   - **Username**: Usually `neo4j`
   - **Password**: The password you set during instance creation
4. Save these credentials securely - you'll need them in Step 4

#### Option B: Local Neo4j (Not recommended for Colab)

If you prefer a local setup, install Neo4j Desktop and ensure it's accessible from your network.

### Step 3: Setting Up OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to **API Keys** section
4. Create a new API key
5. Copy and save the API key securely
6. **Important**: Ensure you have sufficient credits in your OpenAI account for API usage

### Step 4: Installing Required Packages

The packages will be automatically installed when you run the scripts, but you can also install them manually in a Colab cell:

```python
!pip install langchain_core
!pip install langchain_community
!pip install langchain
!pip install langchain_openai
!pip install langchain_experimental
!pip install neo4j
!pip install yfiles_jupyter_graphs
!pip install PyPDF2
!pip install pypdf
!pip install requests
!pip install openai
```

**Note**: The scripts already contain installation commands, so this step is optional.

---

## Project Structure

### 1. Data Collection (`Data Collection.py`)

This script:
- Retrieves research articles from **Semantic Scholar API** related to gestational diabetes mellitus
- Searches for articles from 2000-2024
- Downloads article metadata and saves to CSV files (one per year)
- Downloads open-access PDF files when available
- Creates a `PDFs` folder containing all downloaded PDFs

**Outputs:**
- Multiple CSV files (e.g., `2000.csv`, `2001.csv`, ..., `2024.csv`)
- `PDFs/` folder containing downloaded research paper PDFs

### 2. Entity Extraction and Knowledge Graph Construction (`Entity Extraction and Construction of Knowledge Graph.py`)

This script:
- **Installs required packages** (if not already installed)
- **Connects to Neo4j database** using provided credentials
- **Loads PDF files** from the `PDFs` folder
- **Extracts entities** using GPT-3.5 Turbo with few-shot learning
- **Builds a knowledge graph** in Neo4j with nodes and relationships
- **Implements GraphRAG** for querying the knowledge graph

**Outputs:**
- Knowledge graph stored in Neo4j database
- Graph visualization (if using Neo4j Browser)

---

## How to Execute

### Step 1: Configure Credentials

Before running the scripts, you need to update the credential placeholders in the code.

#### In `Entity Extraction and Construction of Knowledge Graph.py`:

Find and replace the following placeholders:

1. **Neo4j Connection** (around line 15-17):
   ```python
   url = "ENTER YOUR NEO4J URL"        # Replace with your Neo4j Aura URL
   username = "ENTER YOUR NEO4J USERNAME"  # Replace with your Neo4j username
   password = "ENTER YOUR NEO4J PASSWORD"  # Replace with your Neo4j password
   ```

2. **OpenAI API Key** (around line 25 and 104):
   ```python
   os.environ['OPENAI_API_KEY']='ENTER YOUR OPEN AI API KEY'  # Replace with your OpenAI API key
   ```

3. **Additional Neo4j connections** (if present later in the file):
   - Search for `"ENTER YOUR NEO4J URL"` and replace all occurrences
   - Search for `"ENTER YOUR OPEN AI API KEY"` and replace all occurrences

### Step 2: Run Data Collection

1. Open `Data Collection.py` in Colab
2. Run all cells or execute the entire script
3. **Expected behavior:**
   - Script will fetch articles from Semantic Scholar API
   - CSV files will be created for each year (2000-2024)
   - PDF files will be downloaded to the `PDFs` folder
   - Progress messages will be displayed

**Note**: This step may take a considerable amount of time depending on the number of articles and your internet connection.

### Step 3: Run Entity Extraction and Knowledge Graph Construction

1. Ensure Step 2 is complete and you have PDF files in the `PDFs` folder
2. Open `Entity Extraction and Construction of Knowledge Graph.py` in Colab
3. **Important**: Make sure all credentials are properly configured (Step 1)
4. Run all cells or execute the entire script
5. **Expected behavior:**
   - Packages will be installed automatically
   - Connection to Neo4j will be established
   - PDF files will be processed one by one
   - Entities and relationships will be extracted and stored in Neo4j
   - Progress messages will be displayed for each processed document

**Note**: 
- Processing time depends on the number of PDFs and API rate limits
- OpenAI API calls will incur costs based on usage
- Monitor your OpenAI account for usage and billing

### Step 4: Verify the Knowledge Graph

1. Open your Neo4j Aura dashboard or Neo4j Browser
2. Run a simple query to verify data:
   ```cypher
   MATCH (n) RETURN n LIMIT 25
   ```
3. Check the number of nodes and relationships:
   ```cypher
   MATCH (n) RETURN count(n) as node_count
   MATCH ()-[r]->() RETURN count(r) as relationship_count
   ```

---

## Expected File Structure

After running the scripts, your Colab environment should contain:

```
/
├── Data Collection.py
├── Entity Extraction and Construction of Knowledge Graph.py
├── README.md
├── 2000.csv
├── 2001.csv
├── ... (CSV files for each year)
├── 2024.csv
└── PDFs/
    ├── [PaperID1].pdf
    ├── [PaperID2].pdf
    └── ... (all downloaded PDFs)
```

---

## Troubleshooting

### Common Issues

1. **Neo4j Connection Error**
   - Verify your Neo4j URL, username, and password are correct
   - Ensure your Neo4j instance is running (for Aura, it should always be running)
   - Check if your IP address is whitelisted (if required by your Neo4j setup)

2. **OpenAI API Error**
   - Verify your API key is correct and active
   - Check if you have sufficient credits in your OpenAI account
   - Ensure you haven't exceeded rate limits

3. **Package Installation Errors**
   - Restart the Colab runtime: `Runtime > Restart runtime`
   - Re-run the installation cells
   - Check for version conflicts

4. **PDF Processing Errors**
   - Ensure PDFs are downloaded successfully in Step 2
   - Check if PDFs are corrupted or password-protected
   - Verify the `PDFs` folder path is correct

5. **Memory Issues**
   - Colab free tier has limited RAM
   - Process PDFs in smaller batches if needed
   - Consider upgrading to Colab Pro for more resources

6. **API Rate Limits**
   - Semantic Scholar API: Free tier has rate limits
   - OpenAI API: Check your tier's rate limits
   - Add delays between API calls if needed

---

## Important Notes

- **API Costs**: Running this project will incur costs for OpenAI API usage. Monitor your usage regularly.
- **Processing Time**: Full execution may take several hours depending on the dataset size.
- **Data Storage**: Ensure you have sufficient storage in Colab (or mount Google Drive for larger datasets).
- **Neo4j Limits**: Free tier Neo4j Aura has storage and connection limits. Monitor your usage.

---

## Next Steps

After successfully building the knowledge graph:

1. Explore the graph using Neo4j Browser or Cypher queries
2. Use the GraphRAG functionality to query the knowledge graph
3. Visualize relationships between entities
4. Extract insights about gestational diabetes interventions

---

## Support

For issues or questions, please refer to:
- [Neo4j Documentation](https://neo4j.com/docs/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [LangChain Documentation](https://python.langchain.com/)

---

## Note

In parallel, we have prototyped a live GDM chatbot based on this paper, where beta-testers will be able to access a public URL to interact with our model and ask domain-specific questions about gestational diabetes. All announcements, deployment steps, and progress logs will be maintained at the GitHub repository.
