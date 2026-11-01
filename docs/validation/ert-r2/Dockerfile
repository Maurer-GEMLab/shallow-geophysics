# Wine for ResIPy's R2.exe. Built for linux/amd64; Docker Desktop emulates it on Apple Silicon.
#   docker build --platform linux/amd64 -t shallowgeo-r2wine docs/validation/ert-r2
FROM --platform=linux/amd64 debian:bookworm-slim
ENV DEBIAN_FRONTEND=noninteractive WINEDEBUG=-all WINEPREFIX=/root/.wine WINEARCH=win64
RUN apt-get update && apt-get install -y --no-install-recommends wine wine64 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# Create the wine prefix once, at build time, not on every run.
RUN wineboot --init && wineserver -w || true
WORKDIR /work
