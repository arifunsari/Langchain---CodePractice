# 1️⃣ Install the library (run once):
# pip install youtube-transcript-api

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

video_id = "v-jPiFqTOsg"  # Video ID from the URL you provided

try:
    # Try to retrieve the English transcript
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    # Combine the timestamped segments into plain text
    transcript = "\n".join(f"{chunk['start']:.1f}s: {chunk['text']}" for chunk in transcript_list)
    print("✅ Transcript fetched successfully!\n")
    print(transcript)
except TranscriptsDisabled:
    print("❌ Transcript is disabled or not available for this video.")
