# 🧾 VeriClaim: AI-Powered Insurance Claim Eligibility Checker

**VeriClaim** is an intelligent, full-stack web application designed to **instantly assess insurance claim eligibility** and **detect potential fraud** using AI and Machine Learning.  
By integrating **FastAPI**, **React**, and **advanced NLP models**, VeriClaim streamlines the claims review process — offering **real-time, explainable decisions** backed by data and policy understanding.

<br>

## 🌟 Key Features

### 🔍 Real-Time Eligibility Decision
- Instantly determines **APPROVED**, **DENIED**, or **REQUIRES REVIEW** for submitted claims.  
- Combines fraud detection, semantic similarity, and rule-based validation.

### 📄 Semantic Policy Matching
- Uses **sentence embeddings** to compare natural language claims with **policy clauses** extracted from uploaded policy PDFs.
- Employs **vector similarity search** to identify the most relevant clause and justification.

### ⚠️ Fraud & Anomaly Detection
- Trained **Isolation Forest model** flags unusual claims using structured data (e.g., claim amount, patient age, claim history).  
- Supports continuous retraining with updated datasets.

### 💬 Transparent Reasoning
- Every decision includes:
  - The **relevant policy clause**
  - The **semantic similarity score**
  - The **fraud risk analysis result**

### 🖥️ Modern Frontend Dashboard
- Built with **React**, featuring a sleek **dark-themed UI**.
- Interactive result visualization for claims adjusters and reviewers.



## 🧠 Tech Stack Overview

| Component | Technology | Description |
|------------|-------------|-------------|
| **Frontend** | React (JavaScript) | Dynamic UI for claim input and visual feedback |
| **Backend/API** | FastAPI (Python) | Handles requests, integrates ML models, and runs policy analysis |
| **Machine Learning** | Scikit-learn, Joblib | Isolation Forest for fraud and anomaly detection |
| **NLP/Embeddings** | Sentence Transformers | Extracts semantic meaning and performs clause similarity checks |
| **Vector Store** | In-Memory / Pinecone (optional) | Stores and queries policy embeddings efficiently |

<br>

## ⚙️ Getting Started

### Prerequisites
Ensure you have the following installed:

- **Python 3.8+**
- **Node.js (v16+)**
- **npm** or **yarn**



### 🧩 1. Clone the Repository

```bash
git clone https://github.com/aadityaguptaaa/vericlaim.git
cd vericlaim
```

---

### 🐍 2. Backend Setup (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate   # (use venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

Start the FastAPI server:
```bash
uvicorn api.main:app --reload --port 8000
```



### 🧠 3. Load Models & Policy Data

Before running the app for the first time, load the fraud detection model and policy embeddings:

```bash
python load_data.py
```

This script:
- Trains the **Isolation Forest** fraud model using a CSV dataset.
- Extracts and embeds clauses from a sample **policy PDF**.
- Uploads vector data into memory or Pinecone (if configured).



### 💻 4. Frontend Setup (React)

In another terminal:

```bash
cd frontend
npm install
npm start
```

Then open your browser at [http://localhost:3000](http://localhost:3000).



## 🧪 Example Workflow

1. Enter claim details such as **claim amount**, **age**, **claim type**, and **incident description**.  
2. VeriClaim analyzes the input through three modules:
   - **Semantic Policy Match**
   - **Fraud Risk Detection**
   - **Eligibility Decision Engine**
3. Results are shown instantly, with:
   - ✅ Final Decision (Approved / Denied / Requires Review)
   - 🧠 Reasoning Summary
   - 📘 Matching Policy Clause
   - 📊 Fraud Risk Probability

<br>

## 🧰 Folder Structure

```
vericlaim/
│
├── backend/
│   ├── api/
│   │   ├── main.py             # FastAPI entry point
│   │   ├── routes/             # API routes for claims, policy, ML
│   │   └── models/             # ML model management
│   ├── utils/
│   ├── load_data.py            # Model + vector setup
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/         # UI components
│   │   ├── pages/              # Routes and pages
│   │   └── services/           # API calls to FastAPI
│   └── package.json
│
└── README.md
```



## 📈 Future Enhancements

- 🔐 Authentication & Role-Based Access for insurers.
- ☁️ Cloud-based vector storage (Pinecone / FAISS backend).
- 🧾 Policy PDF upload & live embedding extraction.
- 📊 Interactive dashboard with analytics and history tracking.

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to fork this repository, open issues, or submit pull requests.

---

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it with attribution.

<br>

## 🧑‍💻 Author

**Aaditya Gupta**  
**Arpit saxena** 
**Aritra Poddar**

📧 [aadityavidit@gmail.com](mailto:aadityavidit@gmail.com)  
🌐 [LinkedIn](https://linkedin.com/in/aadityaxgupta)



### ⭐ If you found this project helpful, don’t forget to star the repo!
