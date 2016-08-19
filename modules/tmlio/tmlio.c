/*******************************************************************************
 *
 * Adapted version of the twisted mass field read routine
 * to ensure the same conventions in the read/write process.
 *
 *
 *
 *******************************************************************************/

#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <errno.h>
#include "global.h"
#include "su3.h"
#include "dml.h"
#include "lime.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "su3fcts.h"
#include "flags.h"
#include "random.h"
#include "lattice.h"
#include "uflds.h"
#include "archive.h"
#include "global.h"
#include "utils.h"


#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)
#define DEBUG_LEVEL_TMLIO 2
#define ALLOW_LARGE_DIFF_TMLIO

static int endian;
static su3_dble *udb;
static int my_rank;

typedef struct
{
  char   date[64];
  char   package_version[32];

  double beta;
  double c2_rec;
  double epsilonbar;
  double kappa;
  double mu;
  double mubar;
  double plaq;

  int    counter;

  long int time;
}
paramsXlfInfo;

typedef struct
{
  int    lx;
  int    ly;
  int    lz;
  int    lt;
  int    prec;
}
paramsIldgFormat;

typedef struct {
  double plaquetteEnergy;
  int gaugeRead;
  DML_Checksum checksum;
  char * xlfInfo;
  char * ildg_data_lfn;
} paramsGaugeInfo;


static void check_machine(void) {
	error_root(sizeof(stdint_t) != 4, 1, "check_machine [archive.c]",
			"Size of a stdint_t integer is not 4");
	error_root(sizeof(double) != 8, 1, "check_machine [archive.c]",
			"Size of a double is not 8");

	endian = endianness();
	error_root(endian == UNKNOWN_ENDIAN, 1, "check_machine [archive.c]",
			"Unkown endianness");
}

static void finalize_gauge_update(const double plaq0) {
	int ie;
	double nplaq, eps, plaq1;
	set_flags(UPDATED_UD);
	ie=check_bc(64.0*DBL_EPSILON);
	 error_root(ie!=1,1,"import_cnfg [archive.c]",
	 "Incompatible boundary conditions");
	copy_bnd_ud();
	ie = 0;
	nplaq = (double) (3*6 * N0 * N1) * (double) (N2 * N3);
	eps = sqrt(nplaq) * DBL_EPSILON;
	plaq1 = plaq_sum_dble(1) / nplaq;
#ifdef ALLOW_LARGE_DIFF_TMLIO
	ie |= (fabs(plaq1 - plaq0) > 1000*eps);
#else
	ie |= (fabs(plaq1 - plaq0) > eps);
#endif
	if(DEBUG_LEVEL_TMLIO>0&&my_rank==0){
	 printf("# Plaquette calculated: %g, expected %g, diff %g\n", plaq1,plaq0, fabs(plaq1-plaq0));
	 if(ie)
		 printf("# Diff too large\n");
	}

	set_bc();
	plaq1 = plaq_sum_dble(1) / nplaq;
#ifdef ALLOW_LARGE_DIFF_TMLIO
	ie |= (fabs(plaq1 - plaq0) > 1000*eps);
#else
	ie |= (fabs(plaq1 - plaq0) > eps);
#endif
	if(DEBUG_LEVEL_TMLIO>0&&my_rank==0){
	 printf("# Plaquette calculated after set_bc: %g, expected %g, diff %g\n", plaq1,plaq0,fabs(plaq1-plaq0));
	 printf("# BC type: %d\n", bc_type());
	 if(ie)
		 printf("# Diff too large\n");
	}
	error_root(ie != 0, 1, "import_cnfg [archive.c]",
			"Incorrect average plaquette");
	if(my_rank==0){
		fflush(stdout);
	}
}



static void construct_reader(LimeReader ** reader, char * filename) {
	FILE *fh;
	int status;
	fh = NULL;
	status = 0;

	if (DEBUG_LEVEL_TMLIO > 0 && my_rank == 0) {
		printf("# Constructing LIME reader for file %s ...\n", filename);
	}

	fh = fopen(filename, "r");
	status = (fh == NULL) ? 1 : 0;

	error_root(status, 1, "[tmlio]",
			"\nUnable to open file for reading.\nPlease verify file existence and access rights.\nUnable to continue.\n");

	*reader = limeCreateReader(fh);

	error_root((*reader == (LimeReader *) NULL), 1, "[tmlio]",
			"\nCould not create reader, unable to continue.\n");

}

static void destruct_reader(LimeReader * reader) {
	FILE *fh;
	fh = NULL;
	fh = reader->fp;
	limeDestroyReader(reader);
	fclose(fh);
}

static void set_default_paramsIldgFormat(paramsIldgFormat* param) {

	param->prec = 64;
	param->lx = N1;
	param->ly = N2;
	param->lz = N3;
	param->lt = N0;

}

static void swap_assign_single2double(void * out_ptr, void * in_ptr, int nmemb) {
	int j;
	char * char_in_ptr, *char_out_ptr;
	double * double_out_ptr;
	float * float_in_ptr;
	float tmp;

	float_in_ptr = (float *) in_ptr;
	double_out_ptr = (double *) out_ptr;
	char_out_ptr = (char *) &tmp;
	for (j = 0; j < nmemb; j++) {
		char_in_ptr = (char *) float_in_ptr;

		char_out_ptr[3] = char_in_ptr[0];
		char_out_ptr[2] = char_in_ptr[1];
		char_out_ptr[1] = char_in_ptr[2];
		char_out_ptr[0] = char_in_ptr[3];
		(*double_out_ptr) = (double) tmp;
		float_in_ptr++;
		double_out_ptr++;
	}
	return;
}

static void assign_double(void * out_ptr, void * in_ptr, int nmemb) {
	memcpy(out_ptr, in_ptr, 8 * nmemb);
	return;
}

static void assign_single2double(void * out_ptr, void * in_ptr, int nmemb) {
	int i;
	float * float_ptr = (float*) in_ptr;
	double * double_ptr = (double*) out_ptr;

	for (i = 0; i < nmemb; i++) {
		(*double_ptr) = (double) (*float_ptr);

		float_ptr++;
		double_ptr++;
	}

}

static void swap_assign_double(void * out_ptr, void * in_ptr, int nmemb) {
	int j;
	char * char_in_ptr, *char_out_ptr;
	double * double_in_ptr, *double_out_ptr;

	double_in_ptr = (double *) in_ptr;
	double_out_ptr = (double *) out_ptr;
	for (j = 0; j < nmemb; j++) {
		char_in_ptr = (char *) double_in_ptr;
		char_out_ptr = (char *) double_out_ptr;

		char_out_ptr[7] = char_in_ptr[0];
		char_out_ptr[6] = char_in_ptr[1];
		char_out_ptr[5] = char_in_ptr[2];
		char_out_ptr[4] = char_in_ptr[3];
		char_out_ptr[3] = char_in_ptr[4];
		char_out_ptr[2] = char_in_ptr[5];
		char_out_ptr[1] = char_in_ptr[6];
		char_out_ptr[0] = char_in_ptr[7];
		double_in_ptr++;
		double_out_ptr++;
	}
	return;
}

static int read_binary_gauge_data(LimeReader * limereader, DML_Checksum * checksum,
		paramsIldgFormat * input) {
	int t, x, y, z, status = 0;
	int latticeSize[4];
	size_t bytes;
	su3_dble tmp[4], tmprec;
	float tmp2[72], tmp2rec[18]; /* 72=3*3*2*4 gauge links at each point*/
	double tick = 0, tock = 0;
	DML_SiteRank rank;
	size_t expected_size;
	size_t point;

	size_t indwrt;
	const size_t convert_index_mu[] = { 1, 2, 3, 0 };
	size_t mu, nu, offs;
	const int nprocs[] = { NPROC0, NPROC1, NPROC2, NPROC3 };
	int np[4];
	int drank, srank, tag;
	MPI_Status st;
	check_machine();
	expected_size = ((size_t) N0 * N1 * N2 * N3 * 4) * sizeof(su3_dble)
			/ (input->prec == 64 ? 1 : 2);
	/* due to old C standard */
	latticeSize[0] = input->lt;
	latticeSize[1] = input->lx;
	latticeSize[2] = input->ly;
	latticeSize[3] = input->lz;

	udb = udfld();

	DML_checksum_init(checksum);
	if (DEBUG_LEVEL_TMLIO > 0) {
		MPI_Barrier(MPI_COMM_WORLD);
		tick = MPI_Wtime();
	}

	bytes = limeReaderBytes(limereader); /* datalength of ildg-binary-data record in bytes */
	if (bytes != expected_size) {
		fprintf(stderr,"Lattice size and precision found in data file do not match those requested at input.\n");
		fprintf(stderr,"Expected LX = %d, LY = %d, LZ = %d, LT = %d, and %s precision.\n",
				input->lx, input->ly, input->lz, input->lt,
				(input->prec == 64 ? "double" : "single"));
		fprintf(stderr,"Expected %lu bytes, found %lu bytes.\n", expected_size,
				bytes);
		fprintf(stderr,"Check input parameters T, L (LX, LY, LZ) and GaugeConfigReadPrecision.\n");
		fflush(stderr);
		return (-3);
	}

	if (input->prec == 32)
		bytes = (size_t) 2 * sizeof(su3_dble);
	else
		bytes = (size_t) 4 * sizeof(su3_dble);

	for (t = 0; t < L0; t++) {
		for (z = 0; z < L3; z++) {
			for (y = 0; y < L2; y++) {
				/* cpr are the cartesian coordinates of the local process */
				point =
						(size_t) cpr[1] * L1
								+ N1
										* ((L2 * (size_t) cpr[2] + y)
												+ N2
														* ((L3 * (size_t) cpr[3]
																+ z)
																+ N3
																		* (L0
																				* (size_t) cpr[0]
																				+ t)));

				limeReaderSeek(limereader, point * bytes, SEEK_SET);

				for (x = 0; x < L1; x++) {
					rank = (DML_SiteRank) (cpr[1] * L1
							+ (((cpr[0] * L0 + t) * NPROC3 * L3 + cpr[3] * L3
									+ z) * NPROC2 * L2 + cpr[2] * L2 + y)
									* ((DML_SiteRank) L1 * NPROC1) + x);

					if (input->prec == 32) {
						status = limeReaderReadData(tmp2, &bytes, limereader);
						DML_checksum_accum(checksum, rank, (char *) tmp2,
								bytes);
					} else {
						status = limeReaderReadData(tmp, &bytes, limereader);
						DML_checksum_accum(checksum, rank, (char *) tmp, bytes);
					}

					if (status < 0 && status != LIME_EOR) {
						fprintf(stderr,
								"[tmlio] LIME read error occurred with status = %d while reading in gauge_read_binary.c!\n",
								status);
						fflush(stderr);
						return (-2);
					}

					for (mu = 0; mu < 4; mu++) {
						indwrt =
								ipt[z + L3 * y + L2 * L3 * x + L1 * L2 * L3 * t];/*ipt[x3+L3*x2+L2*L3*x1+L1*L2*L3*x0]*/
						nu = convert_index_mu[mu];
						offs = 0;
						if (indwrt < VOLUME / 2) {
							indwrt = iup[indwrt][nu];
							offs = 1;
						}
						if (indwrt >= VOLUME / 2 && indwrt < VOLUME) {
							indwrt = 8 * (indwrt - VOLUME / 2) + offs;
							if (input->prec == 32) {
								if (endian == BIG_ENDIAN) {
									assign_single2double(&udb[indwrt + 2 * nu],
											&tmp2[mu * 18],
											sizeof(su3_dble) / 8);
								} else {
									swap_assign_single2double(
											&udb[indwrt + 2 * nu],
											&tmp2[mu * 18],
											sizeof(su3_dble) / 8);
								}
							} else {
								if (endian == BIG_ENDIAN) {
									assign_double(&udb[indwrt + 2 * nu],
											&tmp[mu], sizeof(su3_dble) / 8);
								} else {
									swap_assign_double(&udb[indwrt + 2 * nu],
											&tmp[mu], sizeof(su3_dble) / 8);
								}

							}/*single*/
						} else { /* in Volume*/
							np[0] = t;
							np[1] = x;
							np[2] = y;
							np[3] = z;
							np[nu] = 0;
							indwrt = ipt[np[3] + L3 * np[2] + L2 * L3 * np[1]
									+ L1 * L2 * L3 * np[0]];
							if (indwrt < VOLUME / 2) {
								/* This should never happen.*/
								fprintf(stderr, "[tmlio]\n Error, point not in volume and not on next process.\n");
							} else {
								/* boundary in direction nu*/
								if (nprocs[nu] < 2) {
									if (input->prec == 32) {
		                                   memcpy(tmp2rec,&tmp2[mu*18],18*sizeof(float));
									}else{
                                    tmprec=tmp[mu];
									}
								} else {
									np[0] = cpr[0];
									np[1] = cpr[1];
									np[2] = cpr[2];
									np[3] = cpr[3];
									np[nu] = (np[nu] + nprocs[nu] - 1)
											% nprocs[nu];
									srank = ipr_global(np);
									np[0] = cpr[0];
									np[1] = cpr[1];
									np[2] = cpr[2];
									np[3] = cpr[3];
									np[nu] = (np[nu] + 1) % nprocs[nu];
									drank = ipr_global(np);
									tag = mpi_tag();
									if (input->prec == 32) {
										MPI_Sendrecv(&tmp2[mu * 18], 18,
												MPI_FLOAT, drank, tag, tmp2rec,
												18, MPI_FLOAT, srank, tag,
												MPI_COMM_WORLD, &st);
									} else {
										MPI_Sendrecv(&tmp[mu], 18, MPI_DOUBLE,
												drank, tag, &tmprec, 18,
												MPI_DOUBLE, srank, tag,
												MPI_COMM_WORLD, &st);
									}/*single*/

								} /*parallel*/
								indwrt = 8 * (indwrt - VOLUME / 2) + offs;
								if (input->prec == 32) {
									if (endian == BIG_ENDIAN) {
										assign_single2double(&udb[indwrt + 2 * nu],
												tmp2rec,
												sizeof(su3_dble) / 8);
									} else {
										swap_assign_single2double(
												&udb[indwrt + 2 * nu],
												tmp2rec,
												sizeof(su3_dble) / 8);
									}
								} else {
									if (endian == BIG_ENDIAN) {
										assign_double(&udb[indwrt + 2 * nu],
												&tmprec, sizeof(su3_dble) / 8);
									} else {
										swap_assign_double(&udb[indwrt + 2 * nu],
												&tmprec, sizeof(su3_dble) / 8);
									}

								}/*single*/
							} /* volume error*/
						}/* in Volume*/
					}/*mu*/
				}/*x*/
			}/*y*/
		}/*z*/
	}/*t*/

	if (DEBUG_LEVEL_TMLIO > 0) {
		MPI_Barrier(MPI_COMM_WORLD);
		tock = MPI_Wtime();

		if (my_rank == 0) {
			printf("# bytes read: %d\n",
					(int) (latticeSize[0] * latticeSize[1] * latticeSize[2]
							* latticeSize[3] * bytes));
			printf("# Time spent reading ");
			printf(" %g s ", tock - tick);
			printf(" %g b/s\n",
					latticeSize[0] * latticeSize[1] * latticeSize[2]
							* latticeSize[3] * bytes / (tock - tick));
			printf("# Reading speed: ");
			printf(" %g b/s",
					latticeSize[0] * latticeSize[1] * latticeSize[2]
							* latticeSize[3] * bytes
							/ (NPROC0 * NPROC1 * NPROC2 * NPROC3 * (tock - tick)));
			printf(" (per MPI process).\n");
		}
	}

	DML_checksum_combine(checksum);
	return (0);
}

static int read_message(LimeReader * reader, char **buffer) {

	int status;
	size_t bytes, bytesRead;
	if (buffer == (char**) NULL)
		return (-1);

	if ((*buffer) != (char*) NULL)
		free(*buffer);
	bytes = limeReaderBytes(reader);
	bytesRead = bytes;

	/* this termination force gives sometimes random results and hanging code ... */
	/* with calloc instead of malloc it seems to be fine                          */
	*buffer = (char*) calloc(bytes + 1, sizeof(char));
	/* *buffer = (char*)calloc(bytes, sizeof(char)); */
	if (*buffer == (char*) NULL) {
		printf("Couldn't malloc data buffer in read_message.\n");
		return (-1);
	}
	status = limeReaderReadData(*buffer, &bytesRead, reader);
	MPI_Barrier(MPI_COMM_WORLD);
	error_root((status != LIME_SUCCESS || bytes != bytesRead), 1, "[tmlio]",
			"Error in reading message.\n");

	(*buffer)[bytes] = '\0'; /* Force termination for safety */
	return (0);
}

static int parse_checksum_xml(char *message, DML_Checksum *checksum) {
	int read_suma = 0, read_sumb = 0;
	char *pos = strtok(message, "<> \n\t");
	if (checksum == (DML_Checksum*) NULL) {
		return 0;
	}
	while (pos) {
		if (!strncmp(pos, "suma", 4)) {
			pos = strtok(0, "<> \n\t");
			sscanf(pos, "%x", &checksum->suma);
			read_suma = 1;
		}
		if (!strncmp(pos, "sumb", 4)) {
			pos = strtok(0, "<> \n\t");
			sscanf(pos, "%x", &checksum->sumb);
			read_sumb = 1;
		}
		pos = strtok(0, "<> \n\t");
	}
	return (read_suma && read_sumb);
}

static int parse_ildgformat_xml(char *message, paramsIldgFormat *ildgformat) {
	int read_prec = 0, read_lx = 0, read_ly = 0, read_lz = 0, read_lt = 0;
	char *pos = strtok(message, "<> \n\t");
	if (ildgformat == (paramsIldgFormat*) NULL) {
		return 0;
	}
	while (pos) {
		if (!strncmp(pos, "precision", 9)) {
			pos = strtok(0, "<> \n\t");
			sscanf(pos, "%d", &ildgformat->prec);
			read_prec = 1;
		}
		if (!strncmp(pos, "lx", 2)) {
			pos = strtok(0, "<> \n\t");
			sscanf(pos, "%d", &ildgformat->lx);
			read_lx = 1;
		}
		if (!strncmp(pos, "ly", 2)) {
			pos = strtok(0, "<> \n\t");
			sscanf(pos, "%d", &ildgformat->ly);
			read_ly = 1;
		}
		if (!strncmp(pos, "lz", 2)) {
			pos = strtok(0, "<> \n\t");
			sscanf(pos, "%d", &ildgformat->lz);
			read_lz = 1;
		}
		if (!strncmp(pos, "lt", 2)) {
			pos = strtok(0, "<> \n\t");
			sscanf(pos, "%d", &ildgformat->lt);
			read_lt = 1;
		}
		pos = strtok(0, "<> \n\t");
	}
	return (read_prec && read_lx && read_ly && read_lz && read_lt);
}

static int parse_Xlf_info(char* message,paramsXlfInfo* params ){
	int read_plaq = 0, read_beta = 0, read_kappa = 0;
	char *pos = strtok(message, "=,\n");
	if (params == (paramsXlfInfo*) NULL) {
		return 0;
	}
	while (pos) {
		if (!strncmp(pos, "plaquette", 9)) {
			pos = strtok(0, "=,\n");
			sscanf(pos, " %lf", &params->plaq);
			read_plaq = 1;
		}
		if (!strncmp(pos, " beta", 4)) {
			pos = strtok(0, "=,\n");
			sscanf(pos, " %lf", &params->beta);
			read_beta = 1;
		}
		if (!strncmp(pos, " kappa", 2)) {
			pos = strtok(0, "=,\n");
			sscanf(pos, " %lf", &params->kappa);
			read_kappa = 1;
		}
		pos = strtok(0, "=,\n");
	}
	return (read_plaq && read_beta && read_kappa);

}

static void close_reader_record(LimeReader *reader) {
	if (reader != NULL)
		limeReaderCloseRecord(reader);
	MPI_Barrier(MPI_COMM_WORLD);
}

static int read_gauge_field(char * filename, paramsGaugeInfo* gaugeinfo) {
	int status = 0;
	char *header_type = NULL;
	LimeReader *reader = NULL; /* either LEMON or LIME Reader*/
	int disable_checks = 0;
	paramsIldgFormat ildgformat_read; /* contains lx,ly,lz,lt,prec*/
	paramsIldgFormat ildgformat_input;
	paramsXlfInfo paramsxlf;
	DML_Checksum checksum_read; /* to uint_32 suma and sumb */
	DML_Checksum checksum_calc;
	int DML_read_flag = 0;
	int gauge_read_flag = 0;
	int gauge_binary_status = 0;
	int ildgformat_read_flag = 0;
	char *checksum_string = NULL;
	char *ildgformat_string = NULL;

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	construct_reader(&reader, filename); /* in case of reading parallel construction of file access handle (with LEMIN MPI file handle) */

	set_default_paramsIldgFormat(&ildgformat_input);/* set defaults accourding to global parameters L, T_global, gauge_precision_read_flag*/


	while ((status = limeReaderNextRecord(reader)) != LIME_EOF) { /* loop through lime records*/

		if (status != LIME_SUCCESS) {
			fprintf(stderr, "[tmlio] limeReaderNextRecord returned status %d.\n", status);
			break;
		}

		header_type = limeReaderType(reader); /* pointer to internal lime value */

		if (header_type == NULL) {
			fprintf(stderr,
					"[tmlio] limeReaderNextRecord returned status %d and header type NULL\n",
					status);
			break;
		}

		if (my_rank == 0 && DEBUG_LEVEL_TMLIO > 2) {
			fprintf(stdout, "found header %s, will now read the message\n",
					header_type);
		}

		if (strcmp("ildg-binary-data", header_type) == 0) {

			/* several times similar way to check whether doublicated entry*/
			if (gauge_read_flag && !disable_checks) { /* a previous ildg-binary-data record has already been read from this file */
				fprintf(stderr,
						"[tmlio] In gauge file %s, multiple LIME records with name: \"ildg-binary-data\" found.\n",
						filename);
				fprintf(stderr,
						"[tmlio] Unable to verify integrity of the gauge field data.\n");
				destruct_reader(reader);
				return (-1);
			}
			/* main reading routine*/
			gauge_binary_status = read_binary_gauge_data(reader, &checksum_calc,
					&ildgformat_input);
			if (gauge_binary_status) {
				fprintf(stderr,
						"[tmlio] Gauge file reading failed at binary part, unable to proceed.\n");
				destruct_reader(reader);
				return (-1);
			}

			gauge_read_flag = 1;
			gaugeinfo->gaugeRead = 1;
			gaugeinfo->checksum = checksum_calc;
		} else if (strcmp("scidac-checksum", header_type) == 0) {
			if (checksum_string == (char*) NULL) {

				/* read and parse xml separated*/
				read_message(reader, &checksum_string);
				DML_read_flag = parse_checksum_xml(checksum_string,
						&checksum_read);
				free(checksum_string);
			} else { /* checksum_string is not NULL, so a scidac-checksum record was already found */
				if (!disable_checks) {
					fprintf(stderr,
							"[tmlio] In gauge file %s, multiple LIME records with name: \"scidac-checksum\" found.\n",
							filename);
					fprintf(stderr,
							"[tmlio] Unable to verify integrity of the gauge field data.\n");
					destruct_reader(reader);
					return (-1);
				}
			}
		} else if (strcmp("xlf-info", header_type) == 0) {

			read_message(reader, &gaugeinfo->xlfInfo);
			parse_Xlf_info(gaugeinfo->xlfInfo,&paramsxlf);
		} else if (strcmp("ildg-data-lfn", header_type) == 0) {

			read_message(reader, &gaugeinfo->ildg_data_lfn);
		} else if (strcmp("ildg-format", header_type) == 0) {

			if (ildgformat_string == (char*) NULL) {
				read_message(reader, &ildgformat_string);
				ildgformat_read_flag = parse_ildgformat_xml(ildgformat_string,
						&ildgformat_read);
				free(ildgformat_string);
			} else { /* ildgformat_string is not NULL, so a ildg-format record was already found */
				if (!disable_checks) {
					fprintf(stderr,
							"[tmlio] In gauge file %s, multiple LIME records with name: \"ildg-format\" found.\n",
							filename);
					fprintf(stderr,
							"[tmlio] Unable to verify integrity of the gauge field data.\n");
					destruct_reader(reader);
					return (-1);
				}
			}
		}

		close_reader_record(reader);
	}

	if (!disable_checks) {

		if (!ildgformat_read_flag) {
			fprintf(stderr,
					"[tmlio] LIME record with name: \"ildg-format\", in gauge file %s either missing or malformed.\n",
					filename);
			fprintf(stderr,
					"[tmlio] Unable to verify gauge field size or precision.\n");
			destruct_reader(reader);
			return (-1);
		}

		if (!gauge_read_flag) {
			fprintf(stderr,
					"[tmlio] LIME record with name: \"ildg-binary-data\", in gauge file %s either missing or malformed.\n",
					filename);
			fprintf(stderr, "[tmlio] No gauge field was read, unable to proceed.\n");
			destruct_reader(reader);
			return (-1);
		}

		if (!DML_read_flag) {
			fprintf(stderr,
					"[tmlio] LIME record with name: \"scidac-checksum\", in gauge file %s either missing or malformed.\n",
					filename);
			fprintf(stderr,
					"[tmlio] Unable to verify integrity of gauge field data.\n");
			destruct_reader(reader);
			return (-1);
		}

		if (my_rank == 0 && DEBUG_LEVEL_TMLIO > 0) {
			/* Verify the integrity of the checksum */
			printf("# Scidac checksums for gaugefield %s:\n", filename);
			printf("#   Calculated            : A = %#010x B = %#010x.\n",
					checksum_calc.suma, checksum_calc.sumb);
			printf("#   Read from LIME headers: A = %#010x B = %#010x.\n",
					checksum_read.suma, checksum_read.sumb);
			fflush(stdout);
		}
		if (checksum_calc.suma != checksum_read.suma) {
			fprintf(stderr,
					"[tmlio] For gauge file %s, calculated and stored values for SciDAC checksum A do not match.\n",
					filename);
			destruct_reader(reader);
			return (-1);
		}
		if (checksum_calc.sumb != checksum_read.sumb) {
			fprintf(stderr,
					"[tmlio] For gauge file %s, calculated and stored values for SciDAC checksum B do not match.\n",
					filename);
			destruct_reader(reader);
			return (-1);
		}

		if (my_rank == 0 && DEBUG_LEVEL_TMLIO > 0) {
			/* Verify the datafile vs the hmc.input parameters */
			printf("# Reading ildg-format record:\n");
			printf("#   Precision = %d bits (%s).\n",
					ildgformat_read.prec,
					(ildgformat_read.prec == 64 ? "double" : "single"));
			printf("#   Lattice size: LX = %d, LY = %d, LZ = %d, LT = %d.\n",
					ildgformat_read.lx, ildgformat_read.ly, ildgformat_read.lz,
					ildgformat_read.lt);
			printf("# Input parameters:\n");
			printf("#   Precision = %d bits (%s).\n",
					ildgformat_input.prec,
					(ildgformat_input.prec == 64 ? "double" : "single"));
			printf("#   Lattice size: LX = %d, LY = %d, LZ = %d, LT = %d.\n",
					ildgformat_input.lx, ildgformat_input.ly,
					ildgformat_input.lz, ildgformat_input.lt);
			printf("#   Parameters: Beta = %g, Kappa = %g\n",
								paramsxlf.beta, paramsxlf.kappa);
		}
	}

	destruct_reader(reader);

	finalize_gauge_update(paramsxlf.plaq);

	return (0);
}

void import_cnfg_tmlio(char *in) {
	paramsGaugeInfo gaugeinfo;
	gaugeinfo.checksum.suma = 0;
	gaugeinfo.checksum.sumb = 0;
	gaugeinfo.gaugeRead = 0;
	gaugeinfo.ildg_data_lfn = NULL;
	gaugeinfo.plaquetteEnergy = 0.0;
	gaugeinfo.xlfInfo = NULL;

	read_gauge_field(in, &gaugeinfo);
	free(gaugeinfo.ildg_data_lfn);
	free(gaugeinfo.xlfInfo);
}
