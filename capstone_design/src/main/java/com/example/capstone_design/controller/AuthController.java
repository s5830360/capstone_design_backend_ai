package com.example.capstone_design.controller;

import com.example.capstone_design.dto.AuthDtos;
import com.example.capstone_design.service.AuthService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;

    @PostMapping("/signup")
    @ResponseStatus(HttpStatus.CREATED)
    public void signup(@RequestBody @Valid AuthDtos.SignupRequest req) {
        authService.signup(req.getUsername(), req.getPassword());
    }

    @PostMapping("/login")
    public AuthDtos.TokenResponse login(@RequestBody @Valid AuthDtos.LoginRequest req) {
        String token = authService.login(req.getUsername(), req.getPassword());
        return new AuthDtos.TokenResponse(token);
    }
}
