#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Matrix); library(data.table); library(SingleCellExperiment)
})
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[1] else Sys.getenv("GLIOMA_ROOT", ".")
pub  <- Sys.getenv("GLIOMA_DATA", file.path(root, "Public_datasets"))
src  <- file.path(pub, "SpaMTP2026_IDHmut_Xenium_MALDI_S-BIAD1426", "processed_data")
out  <- file.path(root, "Project2_spatial_metabolic_niche", "02_processed", "sbiad1426_export")
for (d in c("pixels", "cells", "cell_spe")) dir.create(file.path(out, d), recursive = TRUE, showWarnings = FALSE)
stopifnot(dir.exists(src))
w <- function(df, f) fwrite(as.data.frame(df), f, sep = "\t", quote = FALSE)
wmm <- function(m, f) { f0 <- sub("\\.gz$", "", f); writeMM(as(m, "CsparseMatrix"), f0); system2("gzip", c("-f", shQuote(f0))) }
done <- function(f) file.exists(file.path(out, f))

## ---------------------------------------------------------------- 1. pixel-level MALDI
if (!done("pixels/intensity.mtx.gz")) {
  cat("reading metab_processed.rds ...\n")
  m <- readRDS(file.path(src, "metab_processed.rds"))
  capture.output({ cat(class(m), dim(m), "\n"); str(m[, c(1:3, (ncol(m) - 7):ncol(m))]) }, file = file.path(out, "structure_metab_processed.txt"))
  is_mz <- grepl("^X[0-9]", colnames(m))
  meta <- m[, !is_mz, drop = FALSE]
  meta$pixel_id <- if ("id" %in% names(meta)) meta$id else rownames(m)
  w(meta, file.path(out, "pixels", "coords.tsv.gz"))
  w(data.table(peak = colnames(m)[is_mz], mz = as.numeric(sub("^X", "", colnames(m)[is_mz]))), file.path(out, "pixels", "peaks.tsv.gz"))
  M <- Matrix(as.matrix(m[, is_mz]), sparse = TRUE)       # pixel x peak
  cat(sprintf("  %d pixels x %d peaks, non-zero fraction %.3f\n", nrow(M), ncol(M), length(M@x) / prod(dim(M))))
  wmm(M, file.path(out, "pixels", "intensity.mtx.gz"))
  rm(m, M); invisible(gc())
}

## ---------------------------------------------------------------- 2. cell level (Xenium genes + metabolites mapped to cells)
if (!done("cells/counts_metab.mtx.gz")) {
  cat("reading unified_spe.rds ...\n")
  u <- readRDS(file.path(src, "unified_spe.rds"))
  capture.output(print(u), file = file.path(out, "structure_unified_spe.txt"))
  is_mz <- grepl("^X[0-9]", rownames(u))
  cd <- as.data.frame(colData(u)); cd$cell_id <- colnames(u)
  for (j in seq_along(cd)) if (is.factor(cd[[j]])) cd[[j]] <- as.character(cd[[j]])
  w(cd, file.path(out, "cells", "coldata.tsv.gz"))
  w(data.table(gene = rownames(u)[!is_mz]), file.path(out, "cells", "genes.tsv.gz"))
  w(data.table(peak = rownames(u)[is_mz], mz = as.numeric(sub("^X", "", rownames(u)[is_mz]))), file.path(out, "cells", "metab_peaks.tsv.gz"))
  wmm(counts(u)[!is_mz, ], file.path(out, "cells", "counts_genes.mtx.gz"))
  wmm(counts(u)[is_mz, ],  file.path(out, "cells", "counts_metab.mtx.gz"))
  for (rd in reducedDimNames(u)) w(cbind(cell_id = colnames(u), as.data.frame(reducedDim(u, rd))), file.path(out, "cells", paste0(tolower(rd), ".tsv.gz")))
  cat(sprintf("  %d cells; %d genes; %d m/z\n", ncol(u), sum(!is_mz), sum(is_mz)))
  rm(u); invisible(gc())
}

## ---------------------------------------------------------------- 3. all Xenium cells (cell_spe; optional)
if (!done("cell_spe/counts.mtx.gz") && file.exists(file.path(src, "cell_spe.rds"))) {
  ok <- requireNamespace("SpatialExperiment", quietly = TRUE)
  if (ok) {
    suppressPackageStartupMessages(library(SpatialExperiment))
    s <- readRDS(file.path(src, "cell_spe.rds"))
    capture.output(print(s), file = file.path(out, "structure_cell_spe.txt"))
    cd <- as.data.frame(colData(s)); cd$cell_id <- colnames(s)
    for (j in seq_along(cd)) if (is.factor(cd[[j]]) || is.list(cd[[j]])) cd[[j]] <- as.character(cd[[j]])
    w(cd, file.path(out, "cell_spe", "coldata.tsv.gz"))
    w(cbind(cell_id = colnames(s), as.data.frame(spatialCoords(s))), file.path(out, "cell_spe", "spatialCoords.tsv.gz"))
    w(data.table(feature = rownames(s)), file.path(out, "cell_spe", "features.tsv.gz"))
    wmm(counts(s), file.path(out, "cell_spe", "counts.mtx.gz"))
    cat(sprintf("  cell_spe: %d features x %d cells\n", nrow(s), ncol(s)))
  } else cat("  [skipped] package SpatialExperiment not installed; cell_spe not exported (not needed)\n")
}
cat("done:", out, "\n")
