#pragma once
#include "Import.hpp"
#include "DataSet.hpp"

#ifdef WINConfig
const string report_path = "H:\\Report\\Semantics\\";
const string model_path = "H:\\Model\\Semantics\\";
const string image_reader = "\"C:\\Program Files (x86)\\2345Soft\\2345Pic\\2345PicEditor.exe\" ";
const int parallel_number = 16;

const DataSet SimpleQuestion("Simple Question", "H:\\Data\\Simple Questions.dataset");
#endif

#ifdef LinuxConfig
const string report_path = "/home/bookman/report";
const string model_path = "/home/bookman/report";
const string image_reader = "\"C:\\Program Files (x86)\\2345Soft\\2345Pic\\2345PicEditor.exe\" ";
const int parallel_number = 16;

const DataSet FB15K(
	"FB15K",
	"/home/bookman/FB15K/train.bin",
	"/home/bookman/FB15K/description.bin",
	"/home/bookman/FB15K/test.bin",
	"/home/bookman/FB15K/dev.bin",
	"/home/bookman/FB15K/words.bin");
#endif