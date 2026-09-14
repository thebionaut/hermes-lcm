"""Directional recovery of historical host persistence projections.

Never equate two live image payloads or identify a result by call ID alone.
Only an exact host text projection with a unique archived origin is accepted.
"""
import json


def screenshot_projection(identity):
    role, content, call_id, calls = identity
    if role != 'tool' or not call_id or not content.startswith('['):
        return None
    try:
        parts = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(parts, list) or not parts:
        return None
    texts = []
    saw_image = False
    for part in parts:
        if not isinstance(part, dict):
            return None
        kind = part.get('type')
        if kind == 'text' and isinstance(part.get('text'), str):
            texts.append(part['text'])
        elif kind in {'image', 'image_url', 'input_image'}:
            texts.append('[screenshot]')
            saw_image = True
        else:
            return None
    if not saw_image:
        return None
    return role, '\n'.join(texts), call_id, calls


def extend_projection_mapping(messages, candidates, identities, active_identity,
                              mapping, checkpoint):
    """Fill holes bounded by already-mapped neighboring source records."""
    image_origins = {}
    for row, identity in zip(candidates, identities):
        projection = screenshot_projection(identity)
        if projection is not None:
            image_origins.setdefault(projection, {}).setdefault(identity, []).append(row['store_id'])
    occupied = set(mapping.values())
    for index, message in enumerate(messages):
        checkpoint('historical host projection')
        if id(message) in mapping:
            continue
        lower = max((mapping[id(m)] for m in messages[:index] if id(m) in mapping), default=0)
        upper = min((mapping[id(m)] for m in messages[index+1:] if id(m) in mapping), default=float('inf'))
        identity = active_identity(message)
        origins = image_origins.get(identity, {})
        if len(origins) == 1:
            options = [sid for ids in origins.values() for sid in ids
                       if lower < sid < upper and sid not in occupied]
            if len(options) == 1:
                mapping[id(message)] = options[0]
                occupied.add(options[0])
                continue
        # Hermes joined consecutive user messages with two newlines before
        # inference, while the host persisted the two original messages.
        # Match the complete joined content, never a substring.
        if (identity[0] != 'user' or identity[2] or identity[3]
                or index + 1 >= len(messages)):
            continue
        following = messages[index+1]
        other = active_identity(following)
        if id(following) in mapping or other[0] != 'user' or other[2] or other[3]:
            continue
        combined = ('user', identity[1] + '\n\n' + other[1], '', '')
        options = [row['store_id'] for row, stored in zip(candidates, identities)
                   if stored == combined and lower < row['store_id'] < upper
                   and row['store_id'] not in occupied]
        if len(options) == 1:
            mapping[id(message)] = mapping[id(following)] = options[0]
            occupied.add(options[0])
    return mapping
