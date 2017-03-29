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
	AscentAdaDeltaPositive solver(0.6, 1e-6);
	solver.logout_parameter();
	Model* model = new JointMFactor(FB15K, 1, 5, 0.001, 2.5, 0.04, solver);
	//model->load("H:\\now.model");
	model->train(10, [&](int epos){if (epos % 10 == 0) model->save("H:\\now.model"); });

	while (true)
	{
		boost::char_separator<char> sep(" \t \"\',.\\?!#%@-;:=+-()[]");
		string strin;
		
		logout.record() << "[Input]";
		getline(cin, strin);
		transform(strin.begin(), strin.end(), strin.begin(), tolower);

		boost::tokenizer<boost::char_separator<char>> token(strin, sep);
		vector<int> entity_description;
		for (auto i = token.begin(); i != token.end(); ++i)
		{
			entity_description.push_back(model->dm.words[*i]);
		}

		int eid = model->infer_entity(entity_description);
		string edes;
		for (auto i = model->dm.description[eid].begin(); i != model->dm.description[eid].end(); ++i)
		{
			edes += model->dm.name[*i] + " ";
		}

		logout.record() << "[Question]" << strin;
		logout.record() << "[Answer]" << edes;
	}

	return 0;
}
