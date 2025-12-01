package com.example.capstone_design.service;

import com.example.capstone_design.entity.UserAccount;
import com.example.capstone_design.repository.UserAccountRepository;
import com.example.capstone_design.security.JwtUtil;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

@Service
@RequiredArgsConstructor
public class AuthService {
    private final UserAccountRepository repo;
    private final PasswordEncoder encoder;
    private final JwtUtil jwt;

    @Transactional
    public void signup(String username, String rawPassword) {
        if (repo.existsByUsername(username)) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "이미 존재하는 사용자명");
        }
        UserAccount u = UserAccount.builder()
                .username(username)
                .passwordHash(encoder.encode(rawPassword))
                .role("USER")
                .enabled(true)
                .build();
        repo.save(u);
    }

    public String login(String username, String rawPassword) {
        UserAccount u = repo.findByUsername(username)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Invalid credentials"));
        if (!encoder.matches(rawPassword, u.getPasswordHash())) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Invalid credentials");
        }
        return jwt.generateToken(u.getUsername(), u.getRole());
    }
}
