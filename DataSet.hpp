#pragma once
#include "Import.hpp"

class DataSet
{
public:
	const string name;
	const string url_knowledge;
	const string url_description;
	const string url_test;
	const string url_valid;
	const string url_word;

public:
	DataSet(	const string name, 
				const string url_knowledge, 
				const string url_description,
				const string url_test,
				const string url_valid,
				const string url_word)
				:name(name), 
				url_knowledge(url_knowledge),
				url_description(url_description),
				url_test(url_test),
				url_valid(url_valid),
				url_word(url_word)
	{
		;
	}
};
