# 🧠 Neural Style Transfer Web App

Welcome to the **Artistic Style Transfer Web App**! 🎨 This project lets you turn your ordinary **images and videos** into stunning artworks using **deep learning**. Whether you're a fan of *Van Gogh*, love abstract *geometric art*, or prefer the smooth textures of *oil painting*, this tool brings that creative magic to your media.

Built with a **Flask + PyTorch backend** and a smooth **React frontend**, this app is fast, user-friendly, and super fun to use.

---

## ✨ Features

* 🖼️ **Image & Video Style Transfer** – Just upload your file and watch it transform.
* 🎨 **Multiple Artistic Styles** – Try out "Geometric Painting," "Oil Painting," or "Van Gogh" styles.
* 🧑‍💻 **Clean and Responsive UI** – Built with React + Tailwind CSS for smooth interaction.
* 🚀 **Efficient Backend** – Powered by Flask and PyTorch with CUDA acceleration if available.
* 🖥️ **GPU-Friendly** – Speed up your processing significantly with GPU support (for video especially).

---

## 🧰 Technologies

**Backend:**

* Python 3
* Flask & Flask-CORS
* PyTorch & Torchvision
* OpenCV (cv2), Pillow

**Frontend:**

* React + Vite
* Tailwind CSS + Framer Motion
* Axios for API calls
* Lucide React (icons)

---

## 🚧 Setup Instructions

### ⚙️ Prerequisites

* Python 3.8+
* Node.js & npm (or yarn)
* CUDA-compatible GPU (optional, but highly recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/RonitNaik122/Artistic-Style-Transfer-using-VGG16-AdaIN.git
cd Artistic-Style-Transfer-using-VGG16-AdaIN
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

#### requirements.txt

```
Flask
Flask-Cors
torch
torchvision
Pillow
opencv-python
numpy
uuid
```

### ⬇️ Download Style Models

Create a directory:

```bash
mkdir models
```

Place the `.pth` files (e.g., `geometric_painting.pth`, `oil_painting.pth`, `van_gogh.pth`) into `backend/models/`

### ▶️ Start Backend

```bash
python app.py
```

Flask server runs at: [http://localhost:3000](http://localhost:3000)

### 3. Frontend Setup

```bash
cd ../frontend
npm install  # or yarn install
```

Create `.env` in `/frontend`:

```
VITE_API_URL=http://localhost:3000
```

Then run:

```bash
npm run dev
```

Your frontend is live at [http://localhost:5173](http://localhost:5173)

---

## 🗂️ Project Structure

```
.
├── backend/
│   ├── app.py
│   ├── style_transfer.py
│   ├── models/
│   ├── uploads/
│   ├── results/
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── services/api.js
│   └── vite.config.js
└── README.md
```

---

## 🌐 API Overview

### POST `/api/process`

Uploads a file and processes it with a selected style.

```json
{
  "file": "your_image.jpg",
  "model": "van_gogh.pth"
}
```

**Returns:**

```json
{
  "success": true,
  "output_url": "/api/results/your_file-output.jpg"
}
```

### GET `/api/results/<filename>`

Returns the styled image or video.

---

## 📷 Example Results

> Add your styled image and video outputs here:

| Styled Video Output |
| ---------------------------- |

https://github.com/user-attachments/assets/04d452bd-513d-4405-b162-0531a8010195


https://github.com/user-attachments/assets/dd11dd69-e482-47b0-9be7-84c3db742ae3


https://github.com/user-attachments/assets/6c15c0ae-69fa-4925-ad11-d940ab9ce4bd

---

## 🤝 Contributing

Pull requests are welcome! Follow these steps:

```bash
git checkout -b feature/your-feature
```

Make your changes, push and open a PR 🚀

---

## 🪪 License

This project is licensed under the [MIT License](LICENSE).

---

Enjoy turning your creativity into code! ✨
