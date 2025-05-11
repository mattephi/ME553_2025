//
// Created by Igor on 2025/05/08.
//

#ifndef ME553_2025_SOLUTIONS_EXERCISE1_20244577_HPP_
#define ME553_2025_SOLUTIONS_EXERCISE1_20244577_HPP_

#define ASSIGNMENT_DEBUG

#include <Eigen/Core>
#include <tinyxml_rai/tinyxml_rai.h>
#include <map>
#include <utility>
#include <cstdio>


/*
 * Some mathematics
 */

inline Eigen::Matrix3d RX(double theta) {
    return (Eigen::Matrix3d() << 1, 0, 0, 0, cos(theta), -sin(theta), 0, sin(theta), cos(theta)).finished();
}

inline Eigen::Matrix3d RY(double theta) {
    return (Eigen::Matrix3d() << cos(theta), 0, sin(theta), 0, 1, 0, -sin(theta), 0, cos(theta)).finished();
}

inline Eigen::Matrix3d RZ(double theta) {
    return (Eigen::Matrix3d() << cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1).finished();
}

inline Eigen::Matrix3d R(Eigen::Vector3d p) {
    return RZ(p[2]) * RY(p[1]) * RX(p[0]);
}

struct Placement {
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    Placement() = default;

    Placement(Eigen::Vector3d p, Eigen::Matrix3d R) : p(std::move(p)), R(std::move(R)) {
    }
};

enum class NodeType {
    NONE,
    LINK,
    REVOLUTE_JOINT,
    PRISMATIC_JOINT,
    FIXED_JOINT,
};

inline NodeType getNodeType(const std::string& name) {
    static const std::unordered_map<std::string, NodeType> m{
        {"none", NodeType::NONE},
        {"link", NodeType::LINK},
        {"revolute", NodeType::REVOLUTE_JOINT},
        {"prismatic", NodeType::PRISMATIC_JOINT},
        {"fixed", NodeType::FIXED_JOINT}
    };
    auto it = m.find(name);
    if (it == m.end()) {
        throw std::invalid_argument("Unknown node type: " + name);
    }
    return it->second;
}

inline std::string nodeTypeToString(NodeType type) {
    static const std::unordered_map<NodeType, std::string> m{
        {NodeType::NONE, "none"},
        {NodeType::LINK, "link"},
        {NodeType::REVOLUTE_JOINT, "revolute_joint"},
        {NodeType::PRISMATIC_JOINT, "prismatic_joint"},
        {NodeType::FIXED_JOINT, "fixed_joint"}
    };
    auto it = m.find(type);
    if (it == m.end()) {
        throw std::invalid_argument("Unknown node type");
    }
    return it->second;
}

struct Node {
    virtual ~Node() = default;
    // Metadata
    std::string name;
    NodeType type = NodeType::NONE;

    // Relations
    std::shared_ptr<Node> parent = nullptr;
    std::vector<std::shared_ptr<Node>> children;

    // Actuation & Transformation
    Eigen::VectorXd q = Eigen::VectorXd::Zero(0);
    Eigen::Vector3d axis = Eigen::Vector3d::Zero();
    Eigen::Vector<double, 6> origin = Eigen::Vector<double, 6>::Zero();

    Placement transform(const Placement& P) const {
        return {P.R * origin.head(3) + P.p, P.R * R(origin.tail(3))};
    }

    virtual Placement actuate(const Placement& P) const {
        return P;
    }

    virtual Eigen::VectorXd setGeneralizedCoordinate(Eigen::VectorXd gc) {
        return gc;
    }

    Placement getFramePlacement(const std::string& wrt) const {
        if (wrt == this->name) {
            return {};
        }
        if (this->parent) {
            Placement P = this->parent->getFramePlacement(wrt);
            P = this->transform(P);
            P = this->actuate(P);
            return P;
        }
        throw std::invalid_argument("Frame not found: " + wrt);
    }
};

class FixedJoint final : public Node {
};

class RevoluteJoint final : public Node {
    Placement actuate(const Placement& P) const override {
        return {P.p, P.R * R(this->q[0] * this->axis)};
    }

    Eigen::VectorXd setGeneralizedCoordinate(Eigen::VectorXd gc) override {
        this->q = gc.head(1);
        return gc.tail(gc.size() - 1);
    }
};

class PrismaticJoint final : public Node {
    Placement actuate(const Placement& P) const override {
        return {P.p + P.R * this->axis * this->q[0], P.R};
    }

    Eigen::VectorXd setGeneralizedCoordinate(Eigen::VectorXd gc) override {
        this->q = gc.head(1);
        return gc.tail(gc.size() - 1);
    }
};

inline std::ostream& operator<<(std::ostream& os, const Node& node) {
    os << "-------------------------------\n";
    os << "Node Type: " << nodeTypeToString(node.type) << "\n";
    os << "Name: " << node.name << "\n";
    os << "Axis: " << node.axis.transpose() << "\n";
    os << "Origin: " << node.origin.transpose() << "\n";
    os << "Parent: " << (node.parent ? node.parent->name : "null") << "\n";
    os << "Children: ";
    for (const auto& child : node.children) {
        os << child->name << " ";
    }
    os << "\n";
    os << "Generalized Coordinate: " << node.q.transpose() << "\n";
    os << "-------------------------------\n";
    return os;
}

inline std::map<std::string, std::shared_ptr<Node>> nodes;
inline std::vector<std::shared_ptr<Node>> sorted_nodes;

inline void parseLinks(raisim::TiXmlElement* root) {
    for (raisim::TiXmlElement* child = root->FirstChildElement();
         child != nullptr; child = child->NextSiblingElement()) {
        if (strcmp(child->Value(), "link") == 0) {
            const std::shared_ptr<Node> link = std::make_shared<Node>();
            link->name = child->Attribute("name");
            link->type = NodeType::LINK;
            nodes[link->name] = link;
            sorted_nodes.emplace_back(link);
        }
    }
}

inline void parseJointOrigin(const std::shared_ptr<Node>& joint, const raisim::TiXmlElement* origin) {
    if (!origin) {
        throw std::invalid_argument("Provided joint origin does not exist.");
    }
    const char* xyz = origin->Attribute("xyz");
    const char* rpy = origin->Attribute("rpy");
    Eigen::Vector<double, 6>& jo = joint->origin;
    if (xyz) {
        std::sscanf(xyz, "%lf %lf %lf", &jo[0], &jo[1], &jo[2]);
    }
    if (rpy) {
        std::sscanf(rpy, "%lf %lf %lf", &jo[3], &jo[4], &jo[5]);
    }
}

inline void parseJointAxis(const std::shared_ptr<Node>& joint, const raisim::TiXmlElement* axis) {
    if (!axis) {
        return;
    }
    const char* xyz = axis->Attribute("xyz");
    Eigen::Vector3d& jxyz = joint->axis;
    if (xyz) {
        std::sscanf(xyz, "%lf %lf %lf", &jxyz[0], &jxyz[1], &jxyz[2]);
    }
}

inline std::shared_ptr<Node> create_joint(const NodeType& type) {
    std::shared_ptr<Node> joint = nullptr;
    if (type == NodeType::NONE) {
        throw std::invalid_argument("Joint type is not specified.");
    }
    if (type == NodeType::LINK) {
        joint = std::make_shared<Node>();
    }
    if (type == NodeType::FIXED_JOINT) {
        joint = std::make_shared<FixedJoint>();
    }
    if (type == NodeType::REVOLUTE_JOINT) {
        joint = std::make_shared<RevoluteJoint>();
    }
    if (type == NodeType::PRISMATIC_JOINT) {
        joint = std::make_shared<PrismaticJoint>();
    }
    return joint;
}

inline void updateRelations(std::shared_ptr<Node> joint, const std::string& parent, const std::string& child) {
    std::shared_ptr<Node> joint_parent = nodes[parent];
    std::shared_ptr<Node> joint_child = nodes[child];
    joint_parent->children.emplace_back(joint);
    joint_child->parent = joint;

    joint->children.emplace_back(joint_child);
    joint->parent = joint_parent;
}

inline void parseJoints(raisim::TiXmlElement* root) {
    for (raisim::TiXmlElement* child = root->FirstChildElement(); child != nullptr; child = child->
         NextSiblingElement()) {
        if (strcmp(child->Value(), "joint") == 0) {
            const char* ctype = child->Attribute("type");
            NodeType joint_type = getNodeType(ctype);
            std::shared_ptr<Node> joint = create_joint(joint_type);
            const char* cname = child->Attribute("name");
            joint->name = cname;
            joint->type = joint_type;
            parseJointOrigin(joint, child->FirstChildElement("origin"));
            parseJointAxis(joint, child->FirstChildElement("axis"));

            const char* parent_name = child->FirstChildElement("parent")->Attribute("link");
            const char* child_name = child->FirstChildElement("child")->Attribute("link");

            updateRelations(joint, parent_name, child_name);

            nodes[joint->name] = joint;
            sorted_nodes.emplace_back(joint);
        }
    }
}

inline void parseUrdfFromPath(const std::string& urdf_path) {
    raisim::TiXmlDocument urdf;
    urdf.LoadFile(urdf_path);
    raisim::TiXmlElement* root = urdf.RootElement();
    parseLinks(root);
    parseJoints(root);
}

inline void setGeneralizedCoordinates(const Eigen::VectorXd& gc) {
    Eigen::VectorXd gc_copy = gc;
    for (const auto& joint : sorted_nodes) {
        gc_copy = joint->setGeneralizedCoordinate(gc_copy);
    }
}

#ifdef ASSIGNMENT_DEBUG

inline void printNodes() {
    for (const auto& [key, node] : nodes) {
        std::cout << *node;
    }
}
#endif

/// do not change the name of the method
inline Eigen::Vector3d getEndEffectorPosition(const Eigen::VectorXd& gc) {
    parseUrdfFromPath(std::string(_MAKE_STR(RESOURCE_DIR)) + "/Panda/panda.urdf");
    setGeneralizedCoordinates(gc);
#ifdef ASSIGNMENT_DEBUG
    printNodes();
    std::cout << nodes["panda_rightfinger_tip"]->getFramePlacement("world").p.transpose();
#endif
    return Eigen::Vector3d::Ones(3);
}

#endif // ME553_2025_SOLUTIONS_EXERCISE1_20244577_HPP_
