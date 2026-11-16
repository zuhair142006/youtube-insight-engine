# YouTube Audience & Campaign Intelligence Engine (Clustering Based)

A YouTube-only Streamlit application for a Machine Learning Project. It analyzes public video performance and available public comments, discovers comment-behaviour segments with K-Means, flags unusual video engagement patterns, and produces evidence-linked marketing suggestions.

> **Scope note:** The uploaded course brief describes a broader social-media campaign engine. This implementation follows the more specific project instructions supplied with the request: **YouTube only**. It does not connect to Instagram, TikTok or Facebook.

## Main features

- **YouTube URL input:** Accepts public video URLs, Shorts URLs, channel `/@handle` links, `/channel/UC...` IDs, `/user/...` URLs and legacy `/c/...` URLs (the last option uses YouTube search and costs more API quota).
- **Official API collection:** Uses YouTube Data API v3 to read public channel uploads, video snippets/statistics and public top-level comment threads.
- **CSV alternative:** Import a videos CSV and a comments CSV separately, or upload a combined file. Downloadable CSV templates are included in the app.
- **Video analytics:** Views, likes, comments, engagement rate, video ranking, thumbnails and video links.
- **Comment NLP:** Topic labels for price, discounts, delivery, availability, purchase interest, product/quality, complaints and positive feedback; sentiment and basic English / Roman Urdu / Urdu-script detection.
- **Audience segmentation:** TF-IDF text features + comment interaction features clustered with scikit-learn K-Means.
- **Anomaly detection:** Isolation Forest checks for unusual engagement patterns when enough videos are available. A robust-rate fallback is used for small datasets.
- **Marketing recommendations:** Suggestions are generated from observed topic shares, sentiment and performance outliers.
- **Exports:** Download video performance, analyzed comments and audience-segment summaries as CSV.

## Project structure

```text
youtube_audience_intelligence/
├── app.py                       # Streamlit interface and navigation
├── requirements.txt             # Python dependencies
├── .env.example                 # YouTube API key template
├── .gitignore
├── README.md
├── run_windows.bat              # Optional Windows launch helper
├── src/
│   ├── __init__.py
│   ├── youtube_api.py            # URL parsing and official YouTube API requests
│   ├── data_loader.py            # CSV validation, cleanup and synthetic demo data
│   ├── nlp.py                    # Comment topics, sentiment and language hints
│   ├── ml_models.py              # K-Means and Isolation Forest
│   └── recommendations.py        # Evidence-based marketing suggestions
└── data/
    ├── sample_videos.csv         # Synthetic example data
    └── sample_comments.csv       # Synthetic example data
```

## 1. Install Python

Use Python 3.10 or newer. On Windows, open PowerShell in this project folder.

```powershell
py --version
```

## 2. Create a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, run this once in the same terminal and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Start the application

```powershell
streamlit run app.py
```

Streamlit opens the application in your browser. You can also paste the local URL printed in the terminal (normally `http://localhost:8501`).

## 5. Connect the YouTube Data API (optional)

1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Create or choose a Google Cloud project.
3. Enable **YouTube Data API v3** for that project.
4. Create an API key in **APIs & Services → Credentials**.
5. Paste the key into the **API key** field in the left sidebar, or copy `.env.example` to `.env` and set `YOUTUBE_API_KEY=your_key`.
6. Restart Streamlit after changing `.env`.

Keep the API key private. Do not upload `.env` or commit the key to GitHub. Restrict the key in Google Cloud Console where practical. API quotas apply; collecting many comments can use quota. Keep the video and comment limits modest for a class demonstration.

### What the API can and cannot return

The application uses the public Data API endpoints. Public video/channel metadata and public top-level comment threads are available where the owner has not restricted them. The Data API does **not** provide private YouTube Studio analytics (for example, unique viewers, watch time or detailed audience demographics) for arbitrary channels. Public like counts can also be unavailable. The app preserves blank like counts in the video table; for aggregate engagement calculations an unavailable like count is treated as zero, so that video's displayed engagement rate is a lower-bound estimate.

This project currently downloads top-level comments only. The `reply_count` field records how many replies the thread reports, but the code does not recursively download every reply.

## 6. Run without an API key (CSV mode)

On the **Data & export** page, download the CSV templates. Fill them with data you have collected through permitted methods, then upload both files.

### Videos CSV

Required columns:

| Column | Meaning |
|---|---|
| `video_id` | YouTube video ID (must match IDs in the comments file) |
| `title` | Video title |
| `views` | Public view count |
| `likes` | Public like count; may be blank if unavailable |
| `comments_count` | Public comment count |

Optional columns include `channel_title`, `published_at`, `thumbnail_url`, `video_url`, `duration`, and `description`.

### Comments CSV

Required columns:

| Column | Meaning |
|---|---|
| `video_id` | ID of the video the comment belongs to |
| `comment_text` | Comment text |

Optional columns include `comment_id`, `published_at`, `like_count`, `reply_count`, and `author`. The app does not need author names for segmentation; they are excluded from the default analysis table.

The app also accepts a combined CSV containing the video fields and a `comment_text` field. For clean results, the two-file format is recommended because video and comment timestamps are different fields.

## 7. Try the built-in demo data

Click **Load demo dataset** on the Home page. The records are generated for demonstration and are explicitly marked **SIMULATED DEMO DATA**. The sample is not measured data from any real YouTube channel. Use it to test the dashboard and ML pipeline before uploading your own data.

## How the models work

### NLP

Each comment receives a topic label, a simple sentiment label, a language hint and a possible-purchase-intent flag. English and Roman Urdu keywords are included, along with Urdu-script detection. These rules are designed for a class prototype, not as a validated production language model.

### K-Means audience segments

The app converts comment text to TF-IDF vectors and adds topic/sentiment/language tags plus a small set of behaviour features (comment length, comment likes, reply count, question cue and purchase-intent cue). K-Means groups similar **comments/behaviour signals**. These clusters should be described as comment-behaviour segments, not verified customer profiles.

### Anomaly detection

Isolation Forest evaluates video scale and rates (log views, like rate, comment rate and overall engagement rate). Unusual videos are labelled high or low relative to the dataset's engagement baseline. A small dataset can produce unstable outliers, so use comparable videos and collect more observations where possible.

### Engagement rate

The dashboard calculates:

```text
Engagement rate (%) = (likes + comments) / views × 100
```

If a video has zero views, its rate is set to 0 to avoid division by zero. If a public like count is unavailable, the app calculates a lower-bound rate using comments only; interpret that video's engagement rate with caution.

## Important limitations for the project report

- A public comment is not proof of a unique viewer. One person may post multiple comments, and many viewers never comment.
- The app cannot infer actual age, gender, income, location or purchase behaviour from public comments.
- Topic, sentiment and language labels are approximate. Roman Urdu spellings vary widely; manually review a sample and report the limitations.
- Isolation Forest flags unusual patterns for investigation; it does not prove why a video performed unusually.
- Recommendations are generated from observed dataset patterns and should be presented as suggestions, not causal conclusions.
- Follow YouTube API Services Terms and any data collection, retention and reuse requirements that apply to your dataset.

## Official API references

- [YouTube Data API v3 – Channels: list](https://developers.google.com/youtube/data/reference/rest/v3/channels/list)
- [YouTube Data API v3 – PlaylistItems: list](https://developers.google.com/youtube/data/reference/rest/v3/playlistItems/list)
- [YouTube Data API v3 – Videos: list](https://developers.google.com/youtube/data/reference/rest/v3/videos/list)
- [YouTube Data API v3 – CommentThreads: list](https://developers.google.com/youtube/data/reference/rest/v3/commentThreads/list)


## 8. Publish the app online with GitHub and Streamlit Community Cloud

1. Upload the **contents** of this project folder to a GitHub repository (keep `app.py` and `requirements.txt` in the repository root).
2. Do not upload `.env`, `.venv`, `__pycache__`, or `.streamlit/secrets.toml`. The included `.gitignore` excludes these local/private files.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
4. Choose **Create app**, select your repository and the `main` branch, and set the app file path to `app.py`.
5. To use one private API key for the deployed app, open the deployed app's **Settings → Secrets** and add:

   ```toml
   YOUTUBE_API_KEY = "your-real-api-key"
   ```

6. Save the secrets and reboot the app if prompted. The server-side key is read from Streamlit Secrets and is not prefilled into the visitor-facing password field. Visitors can still use CSV or demo mode without a key.

The GitHub repository stores your source code; Streamlit Community Cloud runs the Streamlit website and gives it a public `streamlit.app` URL. See the official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) for current interface details.
