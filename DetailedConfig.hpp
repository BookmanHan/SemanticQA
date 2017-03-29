#pragma once
#include "Import.hpp"
#include "DataSet.hpp"

const string report_path = "H:\\Report\\Semantics\\";
const string model_path = "H:\\Model\\Semantics\\";
const string image_reader = "\"C:\\Program Files (x86)\\2345Soft\\2345Pic\\2345PicEditor.exe\" ";
const int parallel_number = 16;

const DataSet FB15K(
	"FB15K",
	"H:\\Data\\Knowledge Embedding\\FB15K\\train.bin",
	"H:\\Data\\Knowledge Embedding\\FB15K\\description.bin",
	"H:\\Data\\Knowledge Embedding\\FB15K\\test.bin",
	"H:\\Data\\Knowledge Embedding\\FB15K\\dev.bin",
	"H:\\Data\\Knowledge Embedding\\FB15K\\words.bin");