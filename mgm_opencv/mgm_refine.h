/* Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>,
 *                     Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>,
 *                     Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>*/
#ifndef MGM_REFINE_H_
#define MGM_REFINE_H_
#include <math.h>

#include "refine.h"

//// EXPOSED FUNCTIONS 

typedef float(*refine_t)(const float*, float*, float*);

//// global table of all the refinement functions
struct refinement_functions{
	refine_t f;
	const char *name;
} global_table_of_refinement_functions[] = {
#define REGISTER_FUNCTIONN(x,xn) {x, xn}
	REGISTER_FUNCTIONN(NULL, "none"),
	REGISTER_FUNCTIONN(VfitMinimum, "vfit"),
	REGISTER_FUNCTIONN(ParabolafitMinimum, "parabola"),
	REGISTER_FUNCTIONN(CubicfitMinimum, "cubic"),
	REGISTER_FUNCTIONN(ParabolafitMinimumOpenCV, "parabolaOCV"),
#undef REGISTER_FUNCTIONN
	{ NULL, "" },
};
int get_refinement_index(const char *name) {
	int r = 0; // default refinemnt none
	int N = sizeof(global_table_of_refinement_functions) / sizeof(refinement_functions);
	for (int i = 0; i < N; i++)
	if (strcmp(name, global_table_of_refinement_functions[i].name) == 0)
		r = i;
	return r;
}



// out and outcost are results (size is width*height)
void subpixel_refinement_sgm(struct costvolume_t &S, float* out, float* outcost, char *refinement, int N)
{
	//int N = out.size();

	int idx = get_refinement_index(refinement);
	refine_t refine = global_table_of_refinement_functions[idx].f;

	if (refine) {
		if (TSGM_DEBUG())
			printf("refinement: using %s\n", global_table_of_refinement_functions[idx].name);
		// subpixel refinement
#pragma omp parallel for
		for (int i = 0; i < N; i++) {
			float minP = out[i];
			float minL = outcost[i];

			// can only interpolate if the neighboring disparities are sampled
			int o = minP;
			if (o - 1 >= S[i].min && o + 2 <= S[i].max) {
				if (!std::isfinite(S[i][o - 1]) || !std::isfinite(S[i][o]) || !std::isfinite(S[i][o + 1]) || !std::isfinite(S[i][o + 2])){
					continue;
				}
				float v[4] = { S[i][o - 1], S[i][o], S[i][o + 1], S[i][o + 2] };
				float deltaX = 0;
				refine(v, &minL, &deltaX);
				minP = o + deltaX;
			}

			out[i] = minP;
			outcost[i] = minL;

		}
	}
}
#endif //MGM_REFINE_H_
