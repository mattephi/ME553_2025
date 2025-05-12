#pragma once

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

inline Eigen::Matrix3d RZYX(Eigen::Vector3d p) {
    return RZ(p[2]) * RY(p[1]) * RX(p[0]);
}

inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
        v.z(), 0, -v.x(),
        -v.y(), v.x(), 0;
    return m;
}

struct Placement {
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    Placement() = default;

    Placement(Eigen::Vector3d p, Eigen::Matrix3d R) : p(std::move(p)), R(std::move(R)) {
    }

    // Eigen::Matrix<double, 6, 6> adj() {
    //     Eigen::Matrix3d p_skew = (Eigen::Matrix3d() << 0, -p[2], p[1], p[2], 0, -p[0], -p[1], p[0], 0).finished();
    //     Eigen::Matrix<double, 6, 6> adj;
    //     adj << R, Eigen::Matrix3d::Zero(), p_skew * R, R;
    //     return adj;
    // }
    Eigen::Matrix<double, 6, 6> adj() {
        Eigen::Matrix3d p_skew = skewSymmetric(p);
        Eigen::Matrix<double, 6, 6> Adj = Eigen::Matrix<double, 6, 6>::Zero();
        Adj.block<3, 3>(0, 0) = R;
        // Adj.block<3, 3>(3, 0) = -R * p_skew;
        Adj.block<3, 3>(3, 0) = p_skew * R;
        Adj.block<3, 3>(3, 3) = R;
        return Adj;
    }
};

enum class NodeType {
    NONE,
    LINK,
    REVOLUTE_JOINT,
    CONTINUOUS_JOINT,
    PRISMATIC_JOINT,
    FREE_JOINT,
    FIXED_JOINT,
};

inline NodeType getNodeType(const std::string& name) {
    static const std::unordered_map<std::string, NodeType> m{
        {"none", NodeType::NONE},
        {"link", NodeType::LINK},
        {"revolute", NodeType::REVOLUTE_JOINT},
        {"continuous", NodeType::CONTINUOUS_JOINT},
        {"prismatic", NodeType::PRISMATIC_JOINT},
        {"free", NodeType::FREE_JOINT},
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
        {NodeType::CONTINUOUS_JOINT, "continuous_joint"},
        {NodeType::PRISMATIC_JOINT, "prismatic_joint"},
        {NodeType::FREE_JOINT, "free_joint"},
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
    int nq = 0;
    int nv = 0;
    Eigen::VectorXd q = Eigen::VectorXd::Zero(0);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(0);
    Eigen::Vector3d axis = Eigen::Vector3d::Zero();
    Eigen::Vector<double, 6> origin = Eigen::Vector<double, 6>::Zero();

    // Dynamics
    double mass = 0.0;
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Identity();
    Eigen::Vector<double, 6> iorigin = Eigen::Vector<double, 6>::Zero();

    Eigen::Matrix<double, 6, 6> spatialIntertia() const {
        Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();
        Eigen::Matrix3d r_skew = skewSymmetric(iorigin.head(3));
        /// TODO: Even though cheetah does not have a rotation, implement it
        // Eigen::Matrix3d R = RZYX(this->iorigin.tail(3));
        I.block<3, 3>(0, 0) = inertia + mass * r_skew * r_skew.transpose();
        I.block<3, 3>(3, 3) = mass * Eigen::Matrix3d::Identity();
        I.block<3, 3>(0, 3) = mass * r_skew;
        I.block<3, 3>(3, 0) = mass * r_skew.transpose();
        return I;
    }

    virtual Eigen::MatrixXd jointTwistMap() {
        return Eigen::MatrixXd::Identity(6, this->nv);
    }

    Placement transform(const Placement& P) const {
        return {P.R * origin.head(3) + P.p, P.R * RZYX(origin.tail(3))};
    }

    virtual Placement actuate(const Placement& P) const {
        return P;
    }

    Eigen::VectorXd setGeneralizedCoordinate(Eigen::VectorXd gc) {
        this->q = gc.head(this->nq);
        return gc.tail(gc.size() - this->nq);
    }

    Eigen::VectorXd setGeneralizedVelocity(Eigen::VectorXd gv) {
        this->v = gv.tail(this->nv);
        return gv.tail(gv.size() - this->nv);
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

    virtual Eigen::Vector3d getFrameVelocityEffect(std::shared_ptr<Node> frame) const {
        return Eigen::Vector3d::Zero();
    }

    Eigen::Vector3d getFrameVelocity(std::shared_ptr<Node> frame) const {
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        if (this->parent) {
            v = this->parent->getFrameVelocity(frame);
        }
        v += this->getFrameVelocityEffect(frame);
        return v;
    }

    virtual Eigen::Vector3d getFrameAngularVelocityEffect(std::shared_ptr<Node> frame) const {
        return Eigen::Vector3d::Zero();
    }

    Eigen::Vector3d getFrameAngularVelocity(std::shared_ptr<Node> frame) const {
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        if (this->parent) {
            v = this->parent->getFrameAngularVelocity(frame);
        }
        v += this->getFrameAngularVelocityEffect(frame);
        return v;
    }
};

struct FixedJoint final : public Node {
};

struct RevoluteJoint final : public Node {
    RevoluteJoint() {
        this->nq = 1;
        this->nv = 1;
    }

    Placement actuate(const Placement& P) const override {
        return {P.p, P.R * RZYX(this->q[0] * this->axis)};
    }

    Eigen::MatrixXd jointTwistMap() override {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, this->nv);
        J.block<3, 1>(0, 0) = this->axis;
        return J;
    }

    Eigen::Vector3d getFrameVelocityEffect(std::shared_ptr<Node> frame) const override {
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        Placement frame_P = frame->getFramePlacement("world");
        Placement this_P = this->getFramePlacement("world");
        Eigen::Vector3d axis = this_P.R * this->axis;
        Eigen::Vector3d p = frame_P.p - this_P.p;
        v = axis.cross(p) * this->v[0];
        return v;
    }

    Eigen::Vector3d getFrameAngularVelocityEffect(std::shared_ptr<Node> frame) const override {
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        Placement frame_P = frame->getFramePlacement("world");
        Placement this_P = this->getFramePlacement("world");
        v = this_P.R * this->axis * this->v[0];
        return v;
    }
};

struct PrismaticJoint final : public Node {
    PrismaticJoint() {
        this->nq = 1;
        this->nv = 1;
    }

    Eigen::MatrixXd jointTwistMap() override {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, this->nv);
        J.block<3, 1>(3, 0) = this->axis;
        return J;
    }

    Placement actuate(const Placement& P) const override {
        return {P.p + P.R * this->axis * this->q[0], P.R};
    }

    Eigen::Vector3d getFrameVelocityEffect(std::shared_ptr<Node> frame) const override {
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        Placement frame_P = frame->getFramePlacement("world");
        Placement this_P = this->getFramePlacement("world");
        Eigen::Vector3d axis = this_P.R * this->axis;
        v = axis * this->v[0];
        return v;
    }
};

struct FreeJoint final : public Node {
    // TODO: implement free joint

    FreeJoint() {
        this->nq = 7;
        this->nv = 6;
    }

    Eigen::MatrixXd jointTwistMap() override {
        Eigen::MatrixXd S = Eigen::MatrixXd::Identity(6, this->nv);
        return S;
    }

    Placement actuate(const Placement& P) const override {
        Eigen::Vector3d pos = this->q.head(3);
        Eigen::Quaterniond quat(this->q[3], this->q[4], this->q[5], this->q[6]);
        Eigen::Matrix3d R = quat.toRotationMatrix();
        Eigen::Matrix3d Rtot = P.R * R;
        Eigen::Vector3d ptot = P.R * pos + P.p;
        return {ptot, Rtot};
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
    os << "Generalized Velocity: " << node.v.transpose() << "\n";
    os << "Mass: " << node.mass << "\n";
    os << "Inertia: \n" << node.inertia << "\n";
    os << "IOrigin: " << node.iorigin.transpose() << "\n";
    os << "-------------------------------\n";
    return os;
}

struct Model {
    Model() = default;

    Model(bool free) : free(free) {
    }

    bool free = false;
    std::map<std::string, std::shared_ptr<Node>> nodes;
    std::vector<std::shared_ptr<Node>> sorted_nodes;

    void parseDynamics(std::shared_ptr<Node> node, raisim::TiXmlElement* elem) {
        raisim::TiXmlElement* inertial = elem->FirstChildElement("inertial");
        if (!inertial) return;
        raisim::TiXmlElement* mass = inertial->FirstChildElement("mass");
        if (mass) {
            const char* mass_value = mass->Attribute("value");
            if (mass_value) {
                node->mass = std::atof(mass_value);
            }
        }
        raisim::TiXmlElement* inertia = inertial->FirstChildElement("inertia");
        if (inertia) {
            const char* ixx = inertia->Attribute("ixx");
            const char* ixy = inertia->Attribute("ixy");
            const char* ixz = inertia->Attribute("ixz");
            const char* iyy = inertia->Attribute("iyy");
            const char* iyz = inertia->Attribute("iyz");
            const char* izz = inertia->Attribute("izz");
            if (ixx && iyy && izz) {
                node->inertia << std::atof(ixx), std::atof(ixy), std::atof(ixz),
                    std::atof(ixy), std::atof(iyy), std::atof(iyz),
                    std::atof(ixz), std::atof(iyz), std::atof(izz);
            }
        }
        raisim::TiXmlElement* origin = inertial->FirstChildElement("origin");
        if (origin) {
            const char* xyz = origin->Attribute("xyz");
            const char* rpy = origin->Attribute("rpy");
            if (xyz) {
                std::sscanf(xyz, "%lf %lf %lf", &node->iorigin[0], &node->iorigin[1], &node->iorigin[2]);
            }
            if (rpy) {
                std::sscanf(rpy, "%lf %lf %lf", &node->iorigin[3], &node->iorigin[4], &node->iorigin[5]);
            }
        }
    }

    void parseLinks(raisim::TiXmlElement* root) {
        for (raisim::TiXmlElement* child = root->FirstChildElement();
             child != nullptr; child = child->NextSiblingElement()) {
            if (strcmp(child->Value(), "link") == 0) {
                const std::shared_ptr<Node> link = std::make_shared<Node>();
                link->name = child->Attribute("name");
                link->type = NodeType::LINK;
                nodes[link->name] = link;
                parseDynamics(link, child);
                if (free) {
                    /// WORLD_LINK -> FREE_JOINT -> LINK
                    std::shared_ptr<Node> world = std::make_shared<Node>();
                    world->name = "world";
                    world->type = NodeType::LINK;
                    std::shared_ptr<Node> free_joint = std::make_shared<FreeJoint>();
                    free_joint->name = "free_joint";
                    free_joint->type = NodeType::FREE_JOINT;
                    free_joint->parent = world;
                    world->children.emplace_back(free_joint);
                    free_joint->children.emplace_back(link);
                    link->parent = free_joint;
                    nodes[world->name] = world;
                    nodes[free_joint->name] = free_joint;
                    sorted_nodes.emplace_back(world);
                    sorted_nodes.emplace_back(free_joint);
                    free = false;
                }
                sorted_nodes.emplace_back(link);
            }
        }
    }

    void parseJointOrigin(const std::shared_ptr<Node>& joint, const raisim::TiXmlElement* origin) {
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

    void parseJointAxis(const std::shared_ptr<Node>& joint, const raisim::TiXmlElement* axis) {
        if (!axis) {
            return;
        }
        const char* xyz = axis->Attribute("xyz");
        Eigen::Vector3d& jxyz = joint->axis;
        if (xyz) {
            std::sscanf(xyz, "%lf %lf %lf", &jxyz[0], &jxyz[1], &jxyz[2]);
        }
    }

    std::shared_ptr<Node> create_joint(const NodeType& type) {
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
        if (type == NodeType::CONTINUOUS_JOINT) {
            joint = std::make_shared<RevoluteJoint>();
        }
        if (type == NodeType::PRISMATIC_JOINT) {
            joint = std::make_shared<PrismaticJoint>();
        }
        return joint;
    }

    void updateRelations(std::shared_ptr<Node> joint, const std::string& parent, const std::string& child) {
        std::shared_ptr<Node> joint_parent = nodes[parent];
        std::shared_ptr<Node> joint_child = nodes[child];
        joint_parent->children.emplace_back(joint);
        joint_child->parent = joint;

        joint->children.emplace_back(joint_child);
        joint->parent = joint_parent;
    }

    void parseJoints(raisim::TiXmlElement* root) {
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

    void parseUrdfFromPath(const std::string& urdf_path) {
        raisim::TiXmlDocument urdf;
        urdf.LoadFile(urdf_path);
        raisim::TiXmlElement* root = urdf.RootElement();
        parseLinks(root);
        parseJoints(root);
    }

    void setGeneralizedCoordinates(const Eigen::VectorXd& gc) {
        Eigen::VectorXd gc_copy = gc;
        for (const auto& joint : sorted_nodes) {
            gc_copy = joint->setGeneralizedCoordinate(gc_copy);
        }
    }

    void setGeneralizedVelocities(const Eigen::VectorXd& gv) {
        Eigen::VectorXd gv_copy = gv;
        for (const auto& joint : sorted_nodes) {
            gv_copy = joint->setGeneralizedVelocity(gv_copy);
        }
    }
};

#ifdef ASSIGNMENT_DEBUG

inline void printNodes(const Model& model) {
    for (const auto& node : model.sorted_nodes) {
        std::cout << *node;
    }
}
#endif

/// do not change the name of the method
// inline Eigen::Vector3d getEndEffectorPosition(const Eigen::VectorXd& gc) {
//     Model model;
//     model.parseUrdfFromPath(std::string(_MAKE_STR(RESOURCE_DIR)) + "/Panda/panda.urdf");
//     model.setGeneralizedCoordinates(gc);
// #ifdef ASSIGNMENT_DEBUG
//     printNodes(model);
//     std::cout << model.nodes["panda_rightfinger_tip"]->getFramePlacement("world").p.transpose();
// #endif
//     return Eigen::Vector3d::Ones(3);
// }

/// do not change the name of the method
// inline Eigen::Vector3d getLinearVelocity(const Eigen::VectorXd& gc, const Eigen::VectorXd& gv) {
//     //////////////////////////
//     ///// Your Code Here /////
//     //////////////////////////
//
//     Model model;
//     model.parseUrdfFromPath(std::string(_MAKE_STR(RESOURCE_DIR)) + "/Panda/panda.urdf");
//     model.setGeneralizedCoordinates(gc);
//     model.setGeneralizedVelocities(gv);
// #ifdef ASSIGNMENT_DEBUG
//     printNodes(model);
// #endif
//     return model.nodes["panda_rightfinger_tip"]->getFrameVelocity(model.nodes["panda_rightfinger_tip"]);
//     /// replace this
// }

/// do not change the name of the method
// inline Eigen::Vector3d getAngularVelocity(const Eigen::VectorXd& gc, const Eigen::VectorXd& gv) {
//     //////////////////////////
//     ///// Your Code Here /////
//     //////////////////////////
//     Model model;
//     model.parseUrdfFromPath(std::string(_MAKE_STR(RESOURCE_DIR)) + "/Panda/panda.urdf");
//     model.setGeneralizedCoordinates(gc);
//     model.setGeneralizedVelocities(gv);
// #ifdef ASSIGNMENT_DEBUG
//     printNodes(model);
// #endif
//     return model.nodes["panda_rightfinger_tip"]->getFrameAngularVelocity(model.nodes["panda_rightfinger_tip"]);
//     /// replace this
// }

/// do not change the name of the method
inline Eigen::MatrixXd getMassMatrix(const Eigen::VectorXd& gc) {
    /// !!!!!!!!!! NO RAISIM FUNCTIONS HERE !!!!!!!!!!!!!!!!!

    Model model(true);
    model.parseUrdfFromPath(std::string(std::string(_MAKE_STR(RESOURCE_DIR))) + "/mini_cheetah/urdf/cheetah.urdf");
    model.setGeneralizedCoordinates(gc);
#ifdef ASSIGNMENT_DEBUG
    printNodes(model);
    std::cout << "FREE_JOINT: " << model.nodes["free_joint"]->getFramePlacement("world").p.transpose() << std::endl;
    std::cout << "LF_JOINT1: " << model.nodes["LF_JOINT1"]->getFramePlacement("world").p.transpose() << std::endl;
    std::cout << "RB_JOINT3: " << model.nodes["RB_JOINT3"]->getFramePlacement("world").p.transpose() << std::endl;
    std::cout << "LB_JOINT2: " << model.nodes["LB_JOINT2"]->getFramePlacement("world").p.transpose() << std::endl;
    std::cout << "adjoint: " << model.nodes["LF_JOINT1"]->getFramePlacement("world").adj() << std::endl;
    {
        std::cout << "------------------\n";
        std::cout << "Local RB3:" << std::endl;
        Eigen::MatrixXd Is = model.nodes["RB3"]->spatialIntertia();
        std::cout << "spatial inertia of " << model.nodes["RB3"]->name << ":\n" << model.nodes["RB3"]->spatialIntertia()
            <<
            std::endl;
        Eigen::MatrixXd S = model.nodes["RB_JOINT3"]->jointTwistMap();
        std::cout << "twist map of " << model.nodes["RB_JOINT3"]->name << ":\n" << model.nodes["RB_JOINT3"]->
            jointTwistMap()
            <<
            std::endl;
        std::cout << "got mass of \n" << S.transpose() * Is * S << std::endl;
        std::cout << "------------------\n";
    }
    {
        std::cout << "------------------\n";
        std::cout << "Global RB3\n";
        Eigen::MatrixXd Is = model.nodes["RB3"]->spatialIntertia();
        Placement rb3_p = model.nodes["RB3"]->getFramePlacement("world");
        Eigen::Matrix<double, 6, 6> rb3_adj = rb3_p.adj();
        std::cout << "spatial inertia of " << model.nodes["RB3"]->name << ":\n" << model.nodes["RB3"]->spatialIntertia()
            <<
            std::endl;
        Eigen::MatrixXd S = model.nodes["RB_JOINT3"]->jointTwistMap();
        std::cout << "twist map of " << model.nodes["RB_JOINT3"]->name << ":\n" << model.nodes["RB_JOINT3"]->
            jointTwistMap()
            <<
            std::endl;
        std::cout << "got mass of \n" << (rb3_adj * S).transpose() * rb3_adj.inverse().transpose() * Is * rb3_adj.
            inverse() *
            rb3_adj * S << std::endl;
    }
    {
        std::cout << "------------------\n";
        std::cout << "Local H1717 and H1718\n";
        Eigen::MatrixXd rb3_Is = model.nodes["RB3"]->spatialIntertia();
        Placement rb3_p = model.nodes["RB3"]->getFramePlacement("RB2");
        Eigen::Matrix<double, 6, 6> rb3_adj = rb3_p.adj();
        // std::cout << "rb3_adj: \n" << rb3_adj << std::endl;
        // std::cout << "rb3_adj_inv: \n" << rb3_adj.inverse() << std::endl;
        // std::cout << "rb3_adj_inv_transpose: \n" << rb3_adj.inverse().transpose() << std::endl;
        Eigen::MatrixXd rb3_Is_rb2 = rb3_adj.inverse().transpose() * rb3_Is * rb3_adj.inverse();
        // Eigen::MatrixXd rb3_Is_rb2 = rb3_Is;
        Eigen::MatrixXd rb2_Is = model.nodes["RB2"]->spatialIntertia();
        Eigen::MatrixXd rb2_comp = rb2_Is + rb3_Is_rb2;
        Eigen::MatrixXd S2 = model.nodes["RB_JOINT2"]->jointTwistMap();
        Eigen::MatrixXd S3 = model.nodes["RB_JOINT3"]->jointTwistMap();
        Eigen::MatrixXd rb2_mass = S2.transpose() * rb2_comp * S2;
        std::cout << "H1718: " << S2.transpose() * rb2_comp * rb3_adj * S3 << std::endl;
        std::cout << "RB2 mass: " << rb2_mass << std::endl;
        // std::cout << "adjoint is: " << rb3_adj << std::endl;
        std::cout << "------------------\n";
    }
#endif
    return Eigen::MatrixXd::Ones(18, 18);
}
