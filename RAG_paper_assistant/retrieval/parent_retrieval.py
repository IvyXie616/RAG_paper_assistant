def retrieve_parent_docs(child_docs, parent_map):
    parent_ids = set()

    for doc in child_docs:
        parent_ids.add(doc.metadata["parent_id"])

    parent_docs = [parent_map[pid] for pid in parent_ids]

    return parent_docs