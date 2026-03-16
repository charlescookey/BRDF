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
	float weight;

	glm::vec3 shColour;
	glm::vec3 albedo;
};

struct DisneyBRDFParamsSimple {
	glm::vec3 baseColor = glm::vec3(0.5f);
	float metallic = 0.0f;
	float roughness = 0.5f;
	float specular = 0.5f;

	void clamp() {
		baseColor = glm::clamp(baseColor, glm::vec3(0.0f), glm::vec3(1.0f));
		metallic = glm::clamp(metallic, 0.0f, 1.0f);
		roughness = glm::clamp(roughness, 0.01f, 1.0f);
		specular = glm::clamp(specular, 0.0f, 1.0f);
	}
};
