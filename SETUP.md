# Group ML Trainer — Environment Setup

## Prerequisites

- Python 3.11+
- Node.js 18+ (for the Dashboard)
- A [Supabase](https://supabase.com) project with Storage enabled

## Environment Variables

Create a `.env` file in the project root with:

```
SUPABASE_URL=https://<your-project-ref>.supabase.co
SUPABASE_SERVICE_KEY=<your-service-role-key>
```

The service-role key is required for storage bucket operations and schema verification. You can find it in your Supabase project under **Settings → API → service_role (secret)**.

> **Do not commit `.env` to version control.** The `.gitignore` should already exclude it.

## Database Setup

The schema is defined in `scripts/bootstrap.sql`. You have two options:

### Option A: Run SQL in the Supabase Dashboard

1. Open your Supabase project → **SQL Editor**
2. Paste the contents of `scripts/bootstrap.sql`
3. Click **Run**

### Option B: Use the Supabase CLI

```bash
supabase db push --db-url "postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres" < scripts/bootstrap.sql
```

### Storage Bucket

Create a `checkpoints` storage bucket in your Supabase project:

1. Go to **Storage** in the Supabase Dashboard
2. Click **New bucket**
3. Name it `checkpoints`, set it to **private**

## Verify Schema

After setting up the database, run the verification script:

```bash
pip install supabase python-dotenv
python scripts/verify_schema.py
```

This checks that all 5 tables (`nodes`, `jobs`, `tasks`, `metrics`, `artifacts`) exist with the expected columns, and that the `checkpoints` storage bucket is present.

## Coordinator

```bash
cd coordinator
pip install -r requirements.txt
uvicorn main:app --reload
```

## Worker

```bash
cd worker
pip install -r requirements.txt
python main.py
```

## Dashboard

```bash
cd dashboard
npm install
npm run dev
```
