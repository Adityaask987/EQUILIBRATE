"""
Enhanced Star-rating system:
 - Weighted influence by rater rating
 - Progressive difficulty scaling
 - Decay to neutral over time
 - Sentiment analysis with Hugging Face integration + fallback
 - Anti-gaming protections (cooldown on repeated ratings from same user to same target)
 - Transparency: full stats and history exposure
 - Privacy hooks: anonymized reports and appeal log
 - Configurable tuning knobs
"""

import math
import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque

# Optional advanced sentiment (requires transformers: pip install transformers torch)
try:
    from transformers import pipeline
    _HAVE_TRANSFORMERS = True
except ImportError:
    _HAVE_TRANSFORMERS = False

# -------------------------
# Simple fallback sentiment
# -------------------------
POS_WORDS = {"good", "great", "awesome", "nice", "love", "excellent", "amazing",
             "like", "well", "fantastic", "brilliant", "happy", "recommend"}
NEG_WORDS = {"bad", "terrible", "hate", "awful", "worst", "dislike",
             "sucks", "stupid", "vulgar", "disgusting", "scam", "fraud"}

def simple_rule_sentiment(text):
    """Return 'positive' / 'negative' / 'neutral' using a tiny rule-based approach."""
    if not text or not isinstance(text, str):
        return "neutral"
    text_l = text.lower()
    score = 0
    for w in POS_WORDS:
        if w in text_l:
            score += 1
    for w in NEG_WORDS:
        if w in text_l:
            score -= 1
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

# -------------------------
# Core Rating System
# -------------------------
class StarRatingSystem:
    def __init__(
        self,
        base_change=0.06,
        min_rating=0.0,
        max_rating=5.0,
        neutral_rating=2.5,
        decay_half_life_days=90,
        advanced_sentiment=False,
        cooldown_days=7,  # anti-gaming: same rater cannot affect target again within cooldown
    ):
        self.base_change = base_change
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.neutral = neutral_rating
        self.decay_half_life_days = decay_half_life_days
        self.cooldown_days = cooldown_days

        # Sentiment pipeline if enabled and available
        self.use_advanced_sentiment = advanced_sentiment and _HAVE_TRANSFORMERS
        if self.use_advanced_sentiment:
            self.sentiment_pipe = pipeline("sentiment-analysis")

        # Ratings store
        self.ratings = {}
        self.meta = defaultdict(lambda: {
            "received_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "history": deque(maxlen=500)
        })

        # Anti-gaming: store last feedback timestamp per (rater,target)
        self.last_feedback = {}

        # Privacy: store appeal logs
        self.appeal_log = []

    # -------------------------
    # Utility helpers
    # -------------------------
    def get_rating(self, user_id):
        return self.ratings.get(user_id, self.neutral)

    def _clamp(self, val):
        return max(self.min_rating, min(self.max_rating, val))

    def _now_iso(self):
        return datetime.now(timezone.utc).isoformat()

    def _cooldown_ok(self, target_user, rater_user):
        """Return True if cooldown has passed or no prior feedback."""
        key = (rater_user, target_user)
        if key not in self.last_feedback:
            return True
        last_time = datetime.fromisoformat(self.last_feedback[key])
        if (datetime.now(timezone.utc) - last_time).days >= self.cooldown_days:
            return True
        return False

    # -------------------------
    # Influence & difficulty
    # -------------------------
    def _influence_weight(self, rater_rating):
        norm = rater_rating / self.max_rating
        return 0.1 + 1.1 * math.sqrt(norm)

    def _difficulty_modifier(self, target_rating):
        frac = target_rating / self.max_rating
        modifier = 1.0 - (frac ** 3)
        return max(0.12, modifier)

    # -------------------------
    # Sentiment handling
    # -------------------------
    def sentiment_from_text(self, text):
        if self.use_advanced_sentiment:
            try:
                res = self.sentiment_pipe(text[:512])[0]
                label = res['label'].lower()
                if "pos" in label:
                    return "positive"
                elif "neg" in label:
                    return "negative"
                else:
                    return "neutral"
            except Exception:
                return simple_rule_sentiment(text)
        else:
            return simple_rule_sentiment(text)

    # -------------------------
    # Feedback processing
    # -------------------------
    def process_feedback(self, target_user, rater_user, feedback_type, comment_text=None):
        # Anti-gaming cooldown
        if not self._cooldown_ok(target_user, rater_user):
            return self.get_rating(target_user)  # No effect

        # Determine polarity
        if feedback_type == "like":
            polarity = "positive"
        elif feedback_type == "dislike":
            polarity = "negative"
        elif feedback_type == "comment":
            polarity = self.sentiment_from_text(comment_text)
        else:
            raise ValueError("feedback_type must be 'like'|'dislike'|'comment'")

        # Ratings
        target_rating = self.get_rating(target_user)
        rater_rating = self.get_rating(rater_user)

        influence = self._influence_weight(rater_rating)
        type_mult = {"like": 0.6, "dislike": 0.8, "comment": 1.0}[feedback_type]
        direction = 1 if polarity == "positive" else -1 if polarity == "negative" else 0

        change = 0.0
        if direction != 0:
            change = self.base_change * influence * type_mult * direction
            change *= self._difficulty_modifier(target_rating)

        # Update rating
        new_rating = self._clamp(target_rating + change)
        self.ratings[target_user] = new_rating

        # Update meta
        m = self.meta[target_user]
        m["received_count"] += 1
        if direction == 1:
            m["positive_count"] += 1
        elif direction == -1:
            m["negative_count"] += 1
        m["last_updated"] = self._now_iso()
        m["history"].append({
            "t": m["last_updated"],
            "rater": rater_user,
            "feedback_type": feedback_type,
            "polarity": polarity,
            "old_rating": target_rating,
            "change": round(change, 4),
            "new_rating": new_rating
        })

        # Record last feedback for anti-gaming
        self.last_feedback[(rater_user, target_user)] = m["last_updated"]

        return new_rating

    def bulk_process(self, feedbacks):
        results = []
        for fb in feedbacks:
            results.append(self.process_feedback(*fb))
        return results

    # -------------------------
    # Decay over time
    # -------------------------
    def apply_decay(self):
        k = math.log(2) / max(1.0, self.decay_half_life_days)
        now = datetime.now(timezone.utc)
        for uid, m in self.meta.items():
            last_dt = datetime.fromisoformat(m["last_updated"])
            days = (now - last_dt).total_seconds() / 86400
            if days <= 0:
                continue
            old_r = self.get_rating(uid)
            factor = math.exp(-k * days)
            decayed = self.neutral + (old_r - self.neutral) * factor
            self.ratings[uid] = self._clamp(decayed)
            m["last_updated"] = now.isoformat()
            m["history"].append({
                "t": now.isoformat(),
                "event": "decay",
                "old_rating": old_r,
                "new_rating": self.ratings[uid]
            })

    # -------------------------
    # Privacy & Appeals
    # -------------------------
    def file_appeal(self, target_user, reason):
        self.appeal_log.append({
            "user": target_user,
            "reason": reason,
            "time": self._now_iso()
        })

    def anonymized_report(self, user_id):
        r = self.get_rating(user_id)
        m = self.meta.get(user_id, {})
        return {
            "rating": r,
            "received_count": m.get("received_count", 0),
            "positive_count": m.get("positive_count", 0),
            "negative_count": m.get("negative_count", 0)
        }

    def full_report(self, user_id):
        return {
            "user": user_id,
            **self.meta.get(user_id, {}),
            "rating": self.get_rating(user_id)
        }

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    sys = StarRatingSystem(advanced_sentiment=False)  # set True if transformers installed

    sys.process_feedback("alice", "bob", "like")
    sys.process_feedback("alice", "charlie", "comment", "I love this content, great job!")
    sys.process_feedback("alice", "dan", "comment", "this is disgusting and awful")

    print("Full report for Alice:", json.dumps(sys.full_report("alice"), indent=2))
    print("Anonymized report:", sys.anonymized_report("alice"))

    sys.file_appeal("alice", "Unfair negative review from spam account")
    print("Appeal log:", sys.appeal_log)
