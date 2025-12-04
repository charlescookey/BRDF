#pragma once


#include <vector>
#include <glm/glm.hpp>

struct BRDFSample {
	int splatIndex;
	glm::vec3 omega_i;
	glm::vec3 omega_o;
	glm::vec3 normal;
	glm::vec3 L_i;
	glm::vec3 L_o;
	float cosTheta;
	glm::vec3 fr;
	float weight;
};