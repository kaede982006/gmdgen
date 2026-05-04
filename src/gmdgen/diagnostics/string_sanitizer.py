import re
import unicodedata

class MetadataSanitizer:
    PROMPT_KEYWORDS = ["system prompt", "developer", "assistant", "Ollama", "instruction", "SectionPlan", "ObjectPlan", "TriggerPlan", "raw_ai_plan", "You are an AI", "JSON plan"]
    
    @staticmethod
    def remove_control_chars(text: str) -> str:
        return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

    @staticmethod
    def normalize_unicode(text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    @classmethod
    def detect_prompt_leak(cls, text: str) -> bool:
        lower_text = text.lower()
        for kw in cls.PROMPT_KEYWORDS:
            if kw.lower() in lower_text:
                return True
        return False

    @staticmethod
    def detect_json_blob(text: str) -> bool:
        # If text has high ratio of { } [ ] " : it might be a JSON dump
        chars = set("{}[]\":")
        count = sum(1 for c in text if c in chars)
        if len(text) > 20 and (count / len(text)) > 0.15:
            return True
        if "{" in text and "}" in text and '"' in text and ":" in text:
            return True
        return False

    @staticmethod
    def detect_base64_like_blob(text: str) -> bool:
        # Long sequence of alphanumeric characters without spaces
        words = text.split()
        for w in words:
            if len(w) > 50 and re.match(r'^[A-Za-z0-9+/=]+$', w):
                return True
        return False

    @staticmethod
    def detect_garbage_text(text: str) -> bool:
        if len(text) > 300: # description shouldn't be too long
            return True
        if len(text.splitlines()) > 5:
            return True
        # Check for highly repeated characters
        if re.search(r'(.)\1{10,}', text):
            return True
        return False

    @classmethod
    def sanitize_description(cls, text: str, fallback: str) -> str:
        text = cls.remove_control_chars(text)
        text = cls.normalize_unicode(text)
        if cls.detect_prompt_leak(text) or cls.detect_json_blob(text) or cls.detect_base64_like_blob(text) or cls.detect_garbage_text(text):
            return fallback
        return text.strip()[:200]

    @staticmethod
    def fallback_description(metadata_context: dict) -> str:
        style = metadata_context.get("style", "Modern")
        difficulty = metadata_context.get("difficulty", "Normal")
        sections = metadata_context.get("section_count", 3)
        return f"Audio-synced Geometry Dash level generated with structured planning. Style: {style}. Difficulty: {difficulty}. Sections: {sections}."
