#pragma once

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

inline Eigen::Matrix3d RAA(const Eigen::Vector3d& axis, double angle) {
    return Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
};

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
    std::shared_ptr<Placement> placement = nullptr;

    // Dynamics
    double mass = 0.0;
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
    Eigen::Vector<double, 6> iorigin = Eigen::Vector<double, 6>::Zero();

    Eigen::Matrix3d inertia_W() {
        Placement COM = {iorigin.head(3), RZYX(iorigin.tail(3))};
        Placement this_W = this->getFramePlacement("world");
        Eigen::Matrix3d inertia_W = this->inertia;
        inertia_W = this_W.R * COM.R * inertia_W * COM.R.transpose() * this_W.R.transpose();
        return inertia_W;
    }

    Eigen::Vector3d origin_W() {
        Placement COM = {iorigin.head(3), RZYX(iorigin.tail(3))};
        Placement this_W = this->getFramePlacement("world");
        Eigen::Vector3d origin_W = this->iorigin.head(3);
        origin_W = this_W.p + this_W.R * COM.R * origin_W;
        return origin_W;
    }

    std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> compositeInertia_W() {
        double thisMass = this->mass;
        Eigen::Vector3d thisCOM = this->origin_W();
        Eigen::Matrix3d inertia = this->inertia_W();

        if (this->children.empty()) {
            return {inertia, thisCOM, thisMass};
        }

        for (const auto& child : this->children) {
            double childMass = std::get<2>(child->compositeInertia_W());
            Eigen::Vector3d childCOM = std::get<1>(child->compositeInertia_W());
            Eigen::Matrix3d childInertia = std::get<0>(child->compositeInertia_W());

            Eigen::Vector3d newCOM = (thisMass * thisCOM + childMass * childCOM) / (thisMass + childMass);
            Eigen::Vector3d r1 = thisCOM - newCOM;
            Eigen::Vector3d r2 = childCOM - newCOM;

            Eigen::Matrix3d newInertia = inertia + childInertia - thisMass * skewSymmetric(r1) * skewSymmetric(r1) -
                childMass * skewSymmetric(r2) * skewSymmetric(r2);
            thisMass += childMass;
            thisCOM = newCOM;
            inertia = newInertia;
        }

        if (this->parent->name == "world") {
            if (this->type == NodeType::FREE_JOINT) {
                Eigen::Vector3d OC = thisCOM - this->getFramePlacement("world").p;
                inertia -= thisMass * skewSymmetric(OC) * skewSymmetric(OC);
            }
            if (this->type == NodeType::FIXED_JOINT) {
                throw std::runtime_error("Not implemented");
            }
        }


        return {inertia, thisCOM, thisMass};
    }

    Eigen::Matrix<double, 6, 6> spatialInertia_W() {
        Placement P = this->getFramePlacement("world");
        double totalMass = std::get<2>(this->compositeInertia_W());
        Eigen::Vector3d com = std::get<1>(this->compositeInertia_W());
        Eigen::Matrix3d inertia = std::get<0>(this->compositeInertia_W());
        Eigen::Vector3d JC = com - P.p;

        Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Zero();
        I.block<3, 3>(0, 0) = totalMass * Eigen::Matrix3d::Identity();
        I.block<3, 3>(0, 3) = -skewSymmetric(JC) * totalMass;
        I.block<3, 3>(3, 0) = skewSymmetric(JC) * totalMass;
        I.block<3, 3>(3, 3) = inertia - totalMass * skewSymmetric(JC) * skewSymmetric(JC);

        if (this->parent->name != "world") {
            I.block<3, 3>(3, 3) = inertia - totalMass * skewSymmetric(JC) * skewSymmetric(JC);
        }
        else {
            I.block<3, 3>(3, 3) = inertia;
        }
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

    Placement getFramePlacement(const std::string& wrt) {
        if (placement) {
            return *placement;
        }
        if (wrt == this->name) {
            this->placement = std::make_shared<Placement>();
            return {};
        }
        if (this->parent) {
            Placement P = this->parent->getFramePlacement(wrt);
            P = this->transform(P);
            P = this->actuate(P);
            this->placement = std::make_shared<Placement>(P);
            return P;
        }
        throw std::invalid_argument("Frame not found: " + wrt);
    }

    virtual Eigen::Vector3d getFrameVelocityEffect(std::shared_ptr<Node> frame) {
        return Eigen::Vector3d::Zero();
    }

    Eigen::Vector3d getFrameVelocity(std::shared_ptr<Node> frame) {
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        if (this->parent) {
            v = this->parent->getFrameVelocity(frame);
        }
        v += this->getFrameVelocityEffect(frame);
        return v;
    }

    virtual Eigen::Vector3d getFrameAngularVelocityEffect(std::shared_ptr<Node> frame) {
        return Eigen::Vector3d::Zero();
    }

    Eigen::Vector3d getFrameAngularVelocity(std::shared_ptr<Node> frame) {
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
        return {P.p, P.R * RAA(this->axis, this->q[0])};
    }

    Eigen::MatrixXd jointTwistMap() override {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, this->nv);
        Eigen::Matrix3d R = this->getFramePlacement("world").R;
        Eigen::Vector3d axis = R * this->axis;
        J.block<3, 1>(3, 0) = axis;
        return J;
    }

    Eigen::Vector3d getFrameVelocityEffect(std::shared_ptr<Node> frame) override {
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        Placement frame_P = frame->getFramePlacement("world");
        Placement this_P = this->getFramePlacement("world");
        Eigen::Vector3d axis = this_P.R * this->axis;
        Eigen::Vector3d p = frame_P.p - this_P.p;
        v = axis.cross(p) * this->v[0];
        return v;
    }

    Eigen::Vector3d getFrameAngularVelocityEffect(std::shared_ptr<Node> frame) override {
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
        Eigen::Matrix3d R = this->getFramePlacement("world").R;
        Eigen::Vector3d axis = R * this->axis;
        J.block<3, 1>(0, 0) = axis;
        return J;
    }

    Placement actuate(const Placement& P) const override {
        return {P.p + P.R * this->axis * this->q[0], P.R};
    }

    Eigen::Vector3d getFrameVelocityEffect(std::shared_ptr<Node> frame) override {
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
    std::vector<std::shared_ptr<Node>> sorted_joints;

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
                    sorted_joints.emplace_back(free_joint);
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
                sorted_joints.emplace_back(joint);
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

    int sumNv(int end) {
        int s = 0;
        for (int i = 0; i < end; i++) {
            s += sorted_joints[i]->nv;
        }
        return s;
    }

    int jointInd(std::shared_ptr<Node> joint) {
        for (int i = 0; i < sorted_joints.size(); i++) {
            if (sorted_joints[i] == joint) {
                return i;
            }
        }
        return -1;
    }

    Eigen::MatrixXd crba() {
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(sumNv(sorted_joints.size()), sumNv(sorted_joints.size()));
        for (int i = 12; i >= 0; --i) {
            std::shared_ptr<Node> joint_i = sorted_joints[i];
            int i_start = sumNv(i);
            Eigen::MatrixXd Si = joint_i->jointTwistMap();
            Eigen::Matrix<double, 6, 6> I = joint_i->spatialInertia_W();
            M.block(i_start, i_start, joint_i->nv, joint_i->nv) = Si.transpose() * I * Si;
            std::shared_ptr<Node> current = joint_i;
            while (current->parent->parent) {
                // j->l->j
                current = current->parent->parent;
                Eigen::MatrixXd Sc = current->jointTwistMap();
                Eigen::VectorXd ic = joint_i->getFramePlacement("world").p - current->getFramePlacement("world").p;
                Eigen::Matrix<double, 6, 6> IC = Eigen::Matrix<double, 6, 6>::Identity();
                IC.block<3, 3>(3, 0) = skewSymmetric(ic);
                int c_start = sumNv(jointInd(current));
                M.block(c_start, i_start, current->nv, joint_i->nv) = Sc.transpose() * IC * I * Si;
            }
        }
        M.triangularView<Eigen::Lower>() = M.transpose().triangularView<Eigen::Lower>();
        return M;
    }
};

/// do not change the name of the method
inline Eigen::MatrixXd getMassMatrix(const Eigen::VectorXd& gc) {
    /// !!!!!!!!!! NO RAISIM FUNCTIONS HERE !!!!!!!!!!!!!!!!!

    Model model(true);
    model.parseUrdfFromPath(std::string(_MAKE_STR(RESOURCE_DIR)) + "/mini_cheetah/urdf/cheetah.urdf");
    model.setGeneralizedCoordinates(gc);
    return model.crba();
}
