#pragma once
#include <fstream>
#include <vector>
#include "BRDFSample.h"
#include "utilities.h"

std::vector<BRDFSample> BRDFSampleList;
std::vector<std::vector<BRDFSample>> BRDFSampleList_vec;


void writeBRDFSamples(const std::string& filename, std::vector<Gaussian>& gaussians) {
	std::ofstream file(filename);
	if (!file.is_open()) return;

	file << "splatIndex,omega_i.x,omega_i.y,omega_i.z,omega_o.x,omega_o.y,omega_o.z,normal.x,normal.y,normal.z,L_i.x,L_i.y,L_i.z,L_o.x,L_o.y,L_o.z,cosTheta,weight,SH.r, SH.g, SH.b, albedo.r,albedo.g,albedo.b\n";

	for (const auto& s : BRDFSampleList) {
		file << s.splatIndex << ","
			<< s.omega_i.x << "," << s.omega_i.y << "," << s.omega_i.z << ","
			<< s.omega_o.x << "," << s.omega_o.y << "," << s.omega_o.z << ","
			<< s.normal.x << "," << s.normal.y << "," << s.normal.z << ","
			<< s.L_i.x << "," << s.L_i.y << "," << s.L_i.z << ","
			<< s.L_o.x << "," << s.L_o.y << "," << s.L_o.z << ","
			<< s.cosTheta << ","
			<< s.weight << ","
			<< s.shColour.x << "," << s.shColour.y << "," << s.shColour.z << ","
			<< gaussians[s.splatIndex].color.r << ","
			<< gaussians[s.splatIndex].color.g << ","
			<< gaussians[s.splatIndex].color.b
			<< "\n";
	}

	file << "\n\n";
	file.close();
	std::cout << "Saved " << BRDFSampleList.size()
		<< " BRDF samples to " << filename << std::endl;
}

void writeBRDFSamplesRays(const std::string& filename) {
	std::ofstream file(filename);
	if (!file.is_open()) return;

	file << "splatIndex,omega_i.x,omega_i.y,omega_i.z,omega_o.x,omega_o.y,omega_o.z,normal.x,normal.y,normal.z,L_i.x,L_i.y,L_i.z,L_o.x,L_o.y,L_o.z,cosTheta,weight\n";

	for (const auto& s : BRDFSampleList) {
		file << s.splatIndex << ","
			<< s.omega_i.x << "," << s.omega_i.y << "," << s.omega_i.z << ","
			<< s.omega_o.x << "," << s.omega_o.y << "," << s.omega_o.z << ","
			<< s.normal.x << "," << s.normal.y << "," << s.normal.z << ","
			<< s.L_i.x << "," << s.L_i.y << "," << s.L_i.z << ","
			<< s.L_o.x << "," << s.L_o.y << "," << s.L_o.z << ","
			<< s.cosTheta << ","
			<< s.weight
			<< "\n";
	}

	file << "\n\n";
	file.close();
	std::cout << "Saved " << BRDFSampleList.size()
		<< " BRDF samples to " << filename << std::endl;
}