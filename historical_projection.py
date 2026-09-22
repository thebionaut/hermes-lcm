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


class ProjectionMap(dict):
    """Keep integer boundary compatibility and every source in a joined turn."""
    def __init__(self, mapping):
        super().__init__(mapping)
        self.source_groups = {}


def source_ids(mapping, message):
    key = id(message)
    if key not in mapping:
        return ()
    return getattr(mapping, 'source_groups', {}).get(key, (mapping[key],))


def extend_projection_mapping(messages, candidates, identities, active_identity,
                              mapping, checkpoint):
    """Fill holes bounded by already-mapped neighboring source records."""
    mapping = ProjectionMap(mapping)
    image_origins = {}
    exact_origins = {}
    for row, identity in zip(candidates, identities):
        exact_origins.setdefault(identity, []).append(row['store_id'])
        projection = screenshot_projection(identity)
        if projection is not None:
            image_origins.setdefault(projection, {}).setdefault(identity, []).append(row['store_id'])
    occupied = set(mapping.values())
    # Only the current message and its immediate successor can gain mappings
    # below. Future unmapped messages therefore have the original suffix
    # boundary; carry the prefix maximum forward as mappings are added.
    suffix_upper = [float("inf")] * (len(messages) + 1)
    for index in range(len(messages) - 1, -1, -1):
        if index % 32 == 0:
            checkpoint("historical host projection boundaries")
        suffix_upper[index] = min(
            suffix_upper[index + 1],
            mapping.get(id(messages[index]), float("inf")),
        )
    lower = 0
    for index, message in enumerate(messages):
        if index:
            lower = max(lower, mapping.get(id(messages[index - 1]), 0))
        checkpoint('historical host projection')
        if id(message) in mapping:
            continue
        upper = suffix_upper[index + 1]
        identity = active_identity(message)
        # Duplicate-count reconciliation conservatively leaves the earliest
        # active copies unmapped when active history contains more identical
        # messages than durable storage. That is normally safe, but periodic
        # host-generated messages (for example standing-goal continuations)
        # can repeat after the one durable copy. If mapped neighbors bound a
        # single exact durable source, its position proves which active copy
        # owns it. Recover only that unique in-order match; later repetitions
        # remain unmapped and are ingested as new suffix messages.
        exact_options = [source for source in exact_origins.get(identity, ())
                         if lower < source < upper and source not in occupied]
        if len(exact_options) == 1:
            mapping[id(message)] = exact_options[0]
            occupied.add(exact_options[0])
            continue
        origins = image_origins.get(identity, {})
        if len(origins) == 1:
            options = [sid for ids in origins.values() for sid in ids
                       if lower < sid < upper and sid not in occupied]
            if len(options) == 1:
                mapping[id(message)] = options[0]
                occupied.add(options[0])
                continue
        # The reverse host transformation also occurs on session resume:
        # several durable user rows become one model-facing user message.
        # Prove the entire consecutive group and retain every source ID.
        if identity[0] == 'user' and not identity[2] and not identity[3] and '\n\n' in identity[1]:
            options = []
            for first, (row, stored) in enumerate(zip(candidates, identities)):
                if not lower < row['store_id'] < upper or row['store_id'] in occupied:
                    continue
                if stored[0] != 'user' or stored[2] or stored[3] or not stored[1]:
                    continue
                if not identity[1].startswith(stored[1]+'\n\n'):
                    continue
                joined = stored[1]
                group = [row['store_id']]
                for later in range(first+1,len(candidates)):
                    following = candidates[later]
                    part = identities[later]
                    if (part[0] != 'user' or part[2] or part[3] or not part[1]
                            or following['store_id'] >= upper or following['store_id'] in occupied):
                        break
                    group.append(following['store_id'])
                    joined += '\n\n'+part[1]
                    if joined == identity[1]:
                        options.append(tuple(group))
                        break
                    if not identity[1].startswith(joined+'\n\n'):
                        break
            if len(options) == 1:
                mapping[id(message)] = options[0][-1]
                mapping.source_groups[id(message)] = options[0]
                occupied.update(options[0])
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
