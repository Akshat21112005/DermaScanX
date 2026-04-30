# Skin Cancer Detection Frontend

Next.js frontend for Hugging Face FastAPI backend.

## Setup

```bash
npm install
npm run dev
```

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=https://jiten-333-Skin_Cancer.hf.space
```

Open:

```text
http://localhost:3000
```

## Backend Requirement

Backend must expose:

```text
POST /predict
```

with `multipart/form-data` field name:

```text
file
```
