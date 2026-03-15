# _*_ coding: utf-8 _*_

"""
    @Time : 2025/1/20 14:07 
    @Author : smile 笑
    @File : kg_create.py
    @desc :
"""


import json
from tqdm import tqdm
from py2neo import Graph, Node, Relationship


def connect_to_neo4j(uri="http://localhost:7474", auth=("neo4j", "password"), name="neo4j"):
    graph = Graph(uri, auth=auth, name=name)
    print("成功连接到 Neo4j 数据库。")
    return graph


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_image_node(graph, image_name):
    node = Node("Image", name=image_name)
    graph.merge(node, "Image", "name")


def create_entity_node(graph, entity_data):
    node = Node(
        entity_data['type'],
        name=entity_data['entity'],
        description=entity_data.get('description', '')
    )
    graph.merge(node, entity_data['type'], 'name')


def create_relationship(graph, relation_data, entities_set):
    entity1_name = relation_data['entity1']
    entity2_name = relation_data['entity2']

    if entity1_name in entities_set and entity2_name in entities_set:
        entity1 = graph.nodes.match(name=entity1_name).first()
        entity2 = graph.nodes.match(name=entity2_name).first()
        if entity1 and entity2:
            rel = Relationship(entity1, relation_data['relation'], entity2)
            graph.merge(rel)
    else:
        missing = [name for name in [entity1_name, entity2_name] if name not in entities_set]
        print(f"警告：关系中引用的实体缺失：{', '.join(missing)}")


def link_image_to_entities(graph, image_name, entities):
    image_node = graph.nodes.match("Image", name=image_name).first()
    if not image_node:
        print(f"警告：找不到图像节点 {image_name}。")
        return

    for entity in entities:
        entity_node = graph.nodes.match(name=entity['entity']).first()
        if entity_node:
            rel = Relationship(image_node, "related_to", entity_node)
            graph.merge(rel)


def validate_data(json_data):
    all_entities = set()
    for item in json_data:
        entities = item.get('qa_instruct', {}).get('entities', [])
        for entity in entities:
            all_entities.add(entity['entity'])
    return all_entities


def create_knowledge_graph(graph, json_data):
    print("创建图像节点...")
    for item in tqdm(json_data, desc="图像节点"):
        image_name = item.get('image_name')
        create_image_node(graph, image_name)

    print("创建实体节点和关系...")
    entities_set = validate_data(json_data)
    for item in tqdm(json_data, desc="实体和关系"):
        qa_instruct = item.get('qa_instruct', {})
        entities = qa_instruct.get('entities', [])
        relations = qa_instruct.get('relations', [])

        for entity in entities:
            create_entity_node(graph, entity)
        for relation in relations:
            create_relationship(graph, relation, entities_set)

    print("链接图像与实体...")
    for item in tqdm(json_data, desc="图像与实体关系"):
        image_name = item.get('image_name')
        entities = item.get('qa_instruct', {}).get('entities', [])
        link_image_to_entities(graph, image_name, entities)


def query_image_relations(graph, image_name, limit=20):
    query = f"""
    MATCH (img:Image {{name: '{image_name}'}})-[r:related_to]->(ent)
    OPTIONAL MATCH (ent)-[rel]->(related_ent)
    WHERE NOT related_ent:Image
    RETURN 
        img.name AS image, 
        ent.name AS entity, 
        labels(ent) AS labels, 
        ent.description AS description,
        type(rel) AS relation_type, 
        related_ent.name AS related_entity
    LIMIT {limit}
    """
    results = graph.run(query).data()

    valid_results = []
    for result in results:
        if result.get("related_entity") and result["related_entity"] != image_name:
            valid_results.append(result)

    return valid_results


def main():
    graph = connect_to_neo4j(uri="http://localhost:7474", auth=("neo4j", "zxc123zxc123.."), name="neo4j")
    graph.delete_all()
    file_path = "./kgData/rocov2_kg_content.json"
    json_data = load_json_file(file_path)
    create_knowledge_graph(graph, json_data)
    print("知识图谱创建完成！")


if __name__ == "__main__":
    main()
    graph = connect_to_neo4j(uri="http://localhost:7474", auth=("neo4j", "zxc123zxc123.."), name="neo4j")
    image_name = "ROCOv2_2023_train_000002"
    results = query_image_relations(graph, image_name)
    print(f"图像 {image_name} 的相关信息：")
    for result in results:
        print(result)



