# Caddy with replace_response module for rewriting Swagger/OpenAPI behind path prefixes.
FROM caddy:2-builder AS builder
RUN xcaddy build --with github.com/caddyserver/replace-response

FROM caddy:2-alpine
COPY --from=builder /usr/bin/caddy /usr/bin/caddy
