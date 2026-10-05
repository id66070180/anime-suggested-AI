"""On-demand public metadata: Jikan v4 first, AniList as a fallback."""
import threading
import time

import requests

_lock = threading.Lock()
_last_request = 0.0
_cache = {}


def _request(method, url, **kwargs):
    global _last_request
    # Conservative pacing for both services; HTTP 429 is surfaced, not retried in a loop.
    with _lock:
        delay = max(0, 2.1 - (time.monotonic() - _last_request))
        if delay:
            time.sleep(delay)
        _last_request = time.monotonic()
        response = requests.request(method, url, timeout=12,
                                    headers={"User-Agent": "AnimeCompass-Educational/1.0"}, **kwargs)
    if response.status_code == 429:
        raise RuntimeError(f"API rate limit; try later (Retry-After: {response.headers.get('Retry-After', 'not supplied')})")
    response.raise_for_status()
    return response.json()


def fetch_details(mal_id):
    mal_id = int(mal_id)
    if mal_id < 1:
        raise ValueError("MAL ID must be positive")
    cached = _cache.get(mal_id)
    if cached and time.monotonic() - cached[0] < 3600:
        return cached[1]
    try:
        data = _request("GET", f"https://api.jikan.moe/v4/anime/{mal_id}")["data"]
        if data["mal_id"] != mal_id:
            raise ValueError("Jikan returned a different MAL ID")
        result = {"source": "Jikan / MyAnimeList", "title": data.get("title_english") or data.get("title"),
                  "synopsis": data.get("synopsis"), "status": data.get("status"), "episodes": data.get("episodes"),
                  "score": data.get("score"), "url": data.get("url")}
    except (requests.RequestException, ValueError, KeyError, RuntimeError) as first_error:
        query = """query ($malId: Int!) {
          Media(idMal: $malId, type: ANIME) {
            idMal title { romaji english } description(asHtml: false)
            status episodes siteUrl
          }
        }"""
        try:
            payload = _request("POST", "https://graphql.anilist.co", json={"query": query, "variables": {"malId": mal_id}})
            data = payload.get("data", {}).get("Media")
            if payload.get("errors") or not data or data.get("idMal") != mal_id:
                raise ValueError("AniList did not return the requested anime")
            result = {"source": "AniList", "title": data["title"].get("english") or data["title"].get("romaji"),
                      "synopsis": data.get("description"), "status": data.get("status"),
                      "episodes": data.get("episodes"), "url": data.get("siteUrl")}
        except (requests.RequestException, ValueError, KeyError, RuntimeError) as second_error:
            raise RuntimeError(f"Jikan: {first_error}; AniList: {second_error}") from second_error
    _cache[mal_id] = (time.monotonic(), result)
    return result
