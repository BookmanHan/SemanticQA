#include "Import.hpp"
#include "File.hpp"
#include "Model.hpp"
#include "SemanticModel.hpp"
#include "Task.hpp"
#include "DataProcess.hpp"
#include <boost/format.hpp>  
#include <boost/tokenizer.hpp>  
#include <boost/algorithm/string.hpp>  

int main(int argc, char* argv[])
{
	//data_preprocess();
	//return 0;

	AscentAdaDeltaPositive solver(0.6, 1e-6);
	solver.logout_parameter();
	Model* model = new JointMFactor(FB15K, 100, 10, 2.5, 0.04, solver);
	//model->load("H:\\now.model");
	model->train(1000, [&](int epos){if (epos % 10 == 0) model->save("H:\\now.model"); });

	while (true)
	{
		boost::char_separator<char> sep(" \t \"\',.\\?!#%@-;:=+-()[]");
		string strin;
		
		logout.record().record() << "[Input]";
		getline(cin, strin);
		transform(strin.begin(), strin.end(), strin.begin(), [=](char c){return tolower(c); });

		boost::tokenizer<boost::char_separator<char>> token(strin, sep);
		vector<int> entity_description;
		for (auto i = token.begin(); i != token.end(); ++i)
		{
			if (model->dm.words.find(*i) != model->dm.words.end())
				entity_description.push_back(model->dm.words[*i]);
		}

		vector<int> eid = model->infer_entity(entity_description);
		for (auto val=eid.begin(); val!=eid.end(); ++val)
		{
			if (*val >= model->dm.n_entity)
				continue;

			string edes;
			for (auto i = model->dm.description[*val].begin(); 
				i != min(model->dm.description[*val].begin() + 5, model->dm.description[*val].end()); 
				++i)
			{
				if (*i <model->dm.n_word)
					edes += model->dm.name[*i] + " ";
			}
			logout.record() << "[Answer]" << edes;
		}
	}

	return 0;
}