# Stage 1: Build
FROM node:20-alpine AS build
WORKDIR /app

RUN npm install -g pnpm@10

# Install dependencies first for layer caching
COPY package.json pnpm-lock.yaml ./
COPY patches/ ./patches/
RUN pnpm install --frozen-lockfile

# Copy source and build
COPY . .

ARG VITE_MAPBOX_TOKEN
ARG VITE_FRONTEND_FORGE_API_URL=
ENV VITE_MAPBOX_TOKEN=${VITE_MAPBOX_TOKEN}
ENV VITE_FRONTEND_FORGE_API_URL=${VITE_FRONTEND_FORGE_API_URL}

RUN pnpm build

# Stage 2: Serve
FROM nginx:alpine

COPY --from=build /app/dist/public /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
