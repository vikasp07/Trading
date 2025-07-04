from youtube_transcript_api import YouTubeTranscriptApi
from utils.logger import log_info, log_error

def get_youtube_transcript(video_id):
    """
    Fetches the full transcript for a public YouTube video and
    writes it to UTF‑8 text file.
    """
    try:
        log_info(f"Fetching YouTube transcript for {video_id}")
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join(entry['text'] for entry in transcript)
        out_path = f"data/osint/{video_id}_transcript.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        log_info(f"Saved transcript to {out_path}")

    except Exception as e:
        log_error(f"Failed to fetch transcript for {video_id}: {e}")
