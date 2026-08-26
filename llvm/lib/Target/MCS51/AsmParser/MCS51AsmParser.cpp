//===---- MCS51AsmParser.cpp - Parse MCS51 assembly to MCInst instructions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCS51.h"
#include "MCS51RegisterInfo.h"
#include "MCTargetDesc/MCS51MCELFStreamer.h"
#include "MCTargetDesc/MCS51MCExpr.h"
#include "MCTargetDesc/MCS51MCTargetDesc.h"
#include "TargetInfo/MCS51TargetInfo.h"

#include "llvm/ADT/APInt.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <array>
#include <optional>
#include <sstream>

#define DEBUG_TYPE "avr-asm-parser"

using namespace llvm;

namespace {
/// Parses MCS51 assembly from a stream.
class MCS51AsmParser : public MCTargetAsmParser {
  MCAsmParser &Parser;
  const MCRegisterInfo *MRI;
  const std::string GENERATE_STUBS = "gs";

public:
  enum MCS51MatchResultTy {
    Match_InvalidRegisterOnTiny = FIRST_TARGET_MATCH_RESULT_TY + 1,
    Match_immediate,
    Match_address,
  };

private:
#define GET_ASSEMBLER_HEADER
#include "MCS51GenAsmMatcher.inc"

  bool matchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  bool parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;
  ParseStatus tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                               SMLoc &EndLoc) override;

  bool parseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  ParseStatus parseDirective(AsmToken DirectiveID) override;

  ParseStatus parseMemriOperand(OperandVector &Operands);

  bool parseOperand(OperandVector &Operands, bool maybeReg);
  int parseRegisterName(MCRegister (*matchFn)(StringRef));
  int parseRegisterName();
  int parseRegister(bool RestoreOnFailure = false);
  bool tryParseRegisterOperand(OperandVector &Operands);
  bool tryParseExpression(OperandVector &Operands, bool IsHash = false);
  bool tryParseRelocExpression(OperandVector &Operands);
  void eatComma();

  unsigned validateTargetOperandClass(MCParsedAsmOperand &Op,
                                      unsigned Kind) override;

  unsigned toDREG(unsigned Reg, unsigned From = MCS51::sub_lo) {
    MCRegisterClass const *Class = &MCS51MCRegisterClasses[MCS51::DREGSRegClassID];
    return MRI->getMatchingSuperReg(Reg, From, Class);
  }

  bool emit(MCInst &Instruction, SMLoc const &Loc, MCStreamer &Out) const;
  bool invalidOperand(SMLoc const &Loc, OperandVector const &Operands,
                      uint64_t const &ErrorInfo);
  bool missingFeature(SMLoc const &Loc, uint64_t const &ErrorInfo);

  ParseStatus parseLiteralValues(unsigned SizeInBytes, SMLoc L);

public:
  MCS51AsmParser(const MCSubtargetInfo &STI, MCAsmParser &Parser,
               const MCInstrInfo &MII, const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, STI, MII), Parser(Parser) {
    MCAsmParserExtension::Initialize(Parser);
    MRI = getContext().getRegisterInfo();

    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

  MCAsmParser &getParser() const { return Parser; }
  AsmLexer &getLexer() const { return Parser.getLexer(); }
};

/// An parsed MCS51 assembly operand.
class MCS51Operand : public MCParsedAsmOperand {
  typedef MCParsedAsmOperand Base;
  enum KindTy { k_Immediate, k_Register, k_Token, k_Memri } Kind;

  /// True if this immediate was spelled with the 8051 '#' prefix, which
  /// distinguishes an immediate (#data) from a direct address.
  bool HashPrefixed = false;

public:
  MCS51Operand(StringRef Tok, SMLoc const &S)
      : Kind(k_Token), Tok(Tok), Start(S), End(S) {}
  MCS51Operand(unsigned Reg, SMLoc const &S, SMLoc const &E)
      : Kind(k_Register), RegImm({Reg, nullptr}), Start(S), End(E) {}
  MCS51Operand(MCExpr const *Imm, SMLoc const &S, SMLoc const &E)
      : Kind(k_Immediate), RegImm({0, Imm}), Start(S), End(E) {}
  MCS51Operand(unsigned Reg, MCExpr const *Imm, SMLoc const &S, SMLoc const &E)
      : Kind(k_Memri), RegImm({Reg, Imm}), Start(S), End(E) {}

  struct RegisterImmediate {
    unsigned Reg;
    MCExpr const *Imm;
  };
  union {
    StringRef Tok;
    RegisterImmediate RegImm;
  };

  SMLoc Start, End;

public:
  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(Kind == k_Register && "Unexpected operand kind");
    assert(N == 1 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    // Add as immediate when possible
    if (!Expr)
      Inst.addOperand(MCOperand::createImm(0));
    else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::createImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(Kind == k_Immediate && "Unexpected operand kind");
    assert(N == 1 && "Invalid number of operands!");

    const MCExpr *Expr = getImm();
    addExpr(Inst, Expr);
  }

  /// Adds the contained reg+imm operand to an instruction.
  void addMemriOperands(MCInst &Inst, unsigned N) const {
    assert(Kind == k_Memri && "Unexpected operand kind");
    assert(N == 2 && "Invalid number of operands");

    Inst.addOperand(MCOperand::createReg(getReg()));
    addExpr(Inst, getImm());
  }

  void addImmCom8Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    // The operand is actually a imm8, but we have its bitwise
    // negation in the assembly source, so twiddle it here.
    const auto *CE = cast<MCConstantExpr>(getImm());
    Inst.addOperand(MCOperand::createImm(~(uint8_t)CE->getValue()));
  }

  bool isImmCom8() const {
    if (!isImm())
      return false;
    const auto *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE)
      return false;
    int64_t Value = CE->getValue();
    return isUInt<8>(Value);
  }

  bool isReg() const override { return Kind == k_Register; }
  bool isImm() const override { return Kind == k_Immediate; }
  bool isToken() const override { return Kind == k_Token; }
  bool isMem() const override { return Kind == k_Memri; }
  bool isMemri() const { return Kind == k_Memri; }

  /// Matches an 8051 '#'-prefixed immediate (#data).
  bool isMCS51Imm() const {
    return Kind == k_Immediate && HashPrefixed;
  }

  /// Matches an 8051 bare direct address (no '#' prefix).
  bool isMCS51Direct() const {
    return Kind == k_Immediate && !HashPrefixed;
  }

  StringRef getToken() const {
    assert(Kind == k_Token && "Invalid access!");
    return Tok;
  }

  MCRegister getReg() const override {
    assert((Kind == k_Register || Kind == k_Memri) && "Invalid access!");

    return RegImm.Reg;
  }

  const MCExpr *getImm() const {
    assert((Kind == k_Immediate || Kind == k_Memri) && "Invalid access!");
    return RegImm.Imm;
  }

  static std::unique_ptr<MCS51Operand> CreateToken(StringRef Str, SMLoc S) {
    return std::make_unique<MCS51Operand>(Str, S);
  }

  static std::unique_ptr<MCS51Operand> CreateReg(unsigned RegNum, SMLoc S,
                                               SMLoc E) {
    return std::make_unique<MCS51Operand>(RegNum, S, E);
  }

  static std::unique_ptr<MCS51Operand> CreateImm(const MCExpr *Val, SMLoc S,
                                               SMLoc E, bool IsHash = false) {
    auto Op = std::make_unique<MCS51Operand>(Val, S, E);
    Op->HashPrefixed = IsHash;
    return Op;
  }

  static std::unique_ptr<MCS51Operand>
  CreateMemri(unsigned RegNum, const MCExpr *Val, SMLoc S, SMLoc E) {
    return std::make_unique<MCS51Operand>(RegNum, Val, S, E);
  }

  void makeToken(StringRef Token) {
    Kind = k_Token;
    Tok = Token;
  }

  void makeReg(unsigned RegNo) {
    Kind = k_Register;
    RegImm = {RegNo, nullptr};
  }

  void makeImm(MCExpr const *Ex) {
    Kind = k_Immediate;
    RegImm = {0, Ex};
  }

  void makeMemri(unsigned RegNo, MCExpr const *Imm) {
    Kind = k_Memri;
    RegImm = {RegNo, Imm};
  }

  SMLoc getStartLoc() const override { return Start; }
  SMLoc getEndLoc() const override { return End; }

  void print(raw_ostream &O, const MCAsmInfo &MAI) const override {
    switch (Kind) {
    case k_Token:
      O << "Token: \"" << getToken() << "\"";
      break;
    case k_Register:
      O << "Register: " << getReg().id();
      break;
    case k_Immediate:
      O << "Immediate: \"";
      MAI.printExpr(O, *getImm());
      O << "\"";
      break;
    case k_Memri: {
      // only manually print the size for non-negative values,
      // as the sign is inserted automatically.
      O << "Memri: \"" << getReg().id() << '+';
      MAI.printExpr(O, *getImm());
      O << "\"";
      break;
    }
    }
    O << "\n";
  }
};

} // end anonymous namespace.

// Auto-generated Match Functions

/// Maps from the set of all register names to a register number.
/// \note Generated by TableGen.
static MCRegister MatchRegisterName(StringRef Name);

/// Maps from the set of all alternative registernames to a register number.
/// \note Generated by TableGen.
static MCRegister MatchRegisterAltName(StringRef Name);

bool MCS51AsmParser::invalidOperand(SMLoc const &Loc,
                                  OperandVector const &Operands,
                                  uint64_t const &ErrorInfo) {
  SMLoc ErrorLoc = Loc;
  char const *Diag = nullptr;

  if (ErrorInfo != ~0U) {
    if (ErrorInfo >= Operands.size()) {
      Diag = "too few operands for instruction.";
    } else {
      MCS51Operand const &Op = (MCS51Operand const &)*Operands[ErrorInfo];

      // TODO: See if we can do a better error than just "invalid ...".
      if (Op.getStartLoc() != SMLoc()) {
        ErrorLoc = Op.getStartLoc();
      }
    }
  }

  if (!Diag) {
    Diag = "invalid operand for instruction";
  }

  return Error(ErrorLoc, Diag);
}

bool MCS51AsmParser::missingFeature(llvm::SMLoc const &Loc,
                                  uint64_t const &ErrorInfo) {
  return Error(Loc, "instruction requires a CPU feature not currently enabled");
}

bool MCS51AsmParser::emit(MCInst &Inst, SMLoc const &Loc, MCStreamer &Out) const {
  Inst.setLoc(Loc);
  Out.emitInstruction(Inst, *STI);

  return false;
}

bool MCS51AsmParser::matchAndEmitInstruction(SMLoc Loc, unsigned &Opcode,
                                             OperandVector &Operands,
                                             MCStreamer &Out,
                                             uint64_t &ErrorInfo,
                                             bool MatchingInlineAsm) {
  MCInst Inst;
  unsigned MatchResult =
      MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm);

  switch (MatchResult) {
  case Match_Success:
    return emit(Inst, Loc, Out);
  case Match_MissingFeature:
    return missingFeature(Loc, ErrorInfo);
  case Match_InvalidOperand:
    return invalidOperand(Loc, Operands, ErrorInfo);
  case Match_MnemonicFail:
    return Error(Loc, "invalid instruction");
  case Match_InvalidRegisterOnTiny:
    return Error(Loc, "invalid register on avrtiny");
  case Match_immediate:
  case Match_address:
    return invalidOperand(Loc, Operands, ErrorInfo);
  default:
    return true;
  }
}

/// Parses a register name using a given matching function.
/// Checks for lowercase or uppercase if necessary.
int MCS51AsmParser::parseRegisterName(MCRegister (*matchFn)(StringRef)) {
  StringRef Name = Parser.getTok().getString();

  int RegNum = matchFn(Name);

  // GCC supports case insensitive register names. Some of the MCS51 registers
  // are all lower case, some are all upper case but non are mixed. We prefer
  // to use the original names in the register definitions. That is why we
  // have to test both upper and lower case here.
  if (RegNum == MCS51::NoRegister) {
    RegNum = matchFn(Name.lower());
  }
  if (RegNum == MCS51::NoRegister) {
    RegNum = matchFn(Name.upper());
  }

  return RegNum;
}

int MCS51AsmParser::parseRegisterName() {
  int RegNum = parseRegisterName(&MatchRegisterName);

  if (RegNum == MCS51::NoRegister)
    RegNum = parseRegisterName(&MatchRegisterAltName);

  return RegNum;
}

int MCS51AsmParser::parseRegister(bool RestoreOnFailure) {
  int RegNum = MCS51::NoRegister;

  if (Parser.getTok().is(AsmToken::Identifier)) {
    // Check for register pair syntax
    if (Parser.getLexer().peekTok().is(AsmToken::Colon)) {
      AsmToken HighTok = Parser.getTok();
      Parser.Lex();
      AsmToken ColonTok = Parser.getTok();
      Parser.Lex(); // Eat high (odd) register and colon

      if (Parser.getTok().is(AsmToken::Identifier)) {
        // Convert lower (even) register to DREG
        RegNum = toDREG(parseRegisterName());
      }
      if (RegNum == MCS51::NoRegister && RestoreOnFailure) {
        getLexer().UnLex(std::move(ColonTok));
        getLexer().UnLex(std::move(HighTok));
      }
    } else {
      RegNum = parseRegisterName();
    }
  }
  return RegNum;
}

bool MCS51AsmParser::tryParseRegisterOperand(OperandVector &Operands) {
  int RegNo = parseRegister();

  if (RegNo == MCS51::NoRegister)
    return true;

  // Reject R0~R15 on avrtiny.
  if (RegNo >= static_cast<int>(MCS51::R0) &&
      RegNo <= static_cast<int>(MCS51::R15) &&
      STI->hasFeature(MCS51::FeatureTinyEncoding))
    return Error(Parser.getTok().getLoc(), "invalid register on avrtiny");

  AsmToken const &T = Parser.getTok();
  Operands.push_back(MCS51Operand::CreateReg(RegNo, T.getLoc(), T.getEndLoc()));
  Parser.Lex(); // Eat register token.

  return false;
}

bool MCS51AsmParser::tryParseExpression(OperandVector &Operands,
                                       bool IsHash) {
  SMLoc S = Parser.getTok().getLoc();

  if (!tryParseRelocExpression(Operands))
    return false;

  if ((Parser.getTok().getKind() == AsmToken::Plus ||
       Parser.getTok().getKind() == AsmToken::Minus) &&
      Parser.getLexer().peekTok().getKind() == AsmToken::Identifier) {
    // Don't handle this case - it should be split into two
    // separate tokens.
    return true;
  }

  // Parse (potentially inner) expression
  MCExpr const *Expression;
  if (getParser().parseExpression(Expression))
    return true;

  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  Operands.push_back(MCS51Operand::CreateImm(Expression, S, E, IsHash));
  return false;
}

bool MCS51AsmParser::tryParseRelocExpression(OperandVector &Operands) {
  bool isNegated = false;
  MCS51MCExpr::VariantKind ModifierKind = MCS51MCExpr::VK_MCS51_None;

  SMLoc S = Parser.getTok().getLoc();

  // Reject the form in which sign comes first. This behaviour is
  // in accordance with avr-gcc.
  AsmToken::TokenKind CurTok = Parser.getLexer().getKind();
  if (CurTok == AsmToken::Minus || CurTok == AsmToken::Plus)
    return true;

  // Check for sign.
  AsmToken tokens[2];
  if (Parser.getLexer().peekTokens(tokens) == 2)
    if (tokens[0].getKind() == AsmToken::LParen &&
        tokens[1].getKind() == AsmToken::Minus)
      isNegated = true;

  // Check if we have a target specific modifier (lo8, hi8, &c)
  if (CurTok != AsmToken::Identifier ||
      Parser.getLexer().peekTok().getKind() != AsmToken::LParen) {
    // Not a reloc expr
    return true;
  }
  StringRef ModifierName = Parser.getTok().getString();
  ModifierKind = MCS51MCExpr::getKindByName(ModifierName);

  if (ModifierKind != MCS51MCExpr::VK_MCS51_None) {
    Parser.Lex();
    Parser.Lex(); // Eat modifier name and parenthesis
    if (Parser.getTok().getString() == GENERATE_STUBS &&
        Parser.getTok().getKind() == AsmToken::Identifier) {
      std::string GSModName = ModifierName.str() + "_" + GENERATE_STUBS;
      ModifierKind = MCS51MCExpr::getKindByName(GSModName);
      if (ModifierKind != MCS51MCExpr::VK_MCS51_None)
        Parser.Lex(); // Eat gs modifier name
    }
  } else {
    return Error(Parser.getTok().getLoc(), "unknown modifier");
  }

  if (tokens[1].getKind() == AsmToken::Minus ||
      tokens[1].getKind() == AsmToken::Plus) {
    Parser.Lex();
    assert(Parser.getTok().getKind() == AsmToken::LParen);
    Parser.Lex(); // Eat the sign and parenthesis
  }

  MCExpr const *InnerExpression;
  if (getParser().parseExpression(InnerExpression))
    return true;

  if (tokens[1].getKind() == AsmToken::Minus ||
      tokens[1].getKind() == AsmToken::Plus) {
    assert(Parser.getTok().getKind() == AsmToken::RParen);
    Parser.Lex(); // Eat closing parenthesis
  }

  // If we have a modifier wrap the inner expression
  assert(Parser.getTok().getKind() == AsmToken::RParen);
  Parser.Lex(); // Eat closing parenthesis

  MCExpr const *Expression =
      MCS51MCExpr::create(ModifierKind, InnerExpression, isNegated, getContext());

  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  Operands.push_back(MCS51Operand::CreateImm(Expression, S, E));

  return false;
}

bool MCS51AsmParser::parseOperand(OperandVector &Operands, bool maybeReg) {
  LLVM_DEBUG(dbgs() << "parseOperand\n");

  switch (getLexer().getKind()) {
  default:
    return Error(Parser.getTok().getLoc(), "unexpected token in operand");

  case AsmToken::Hash:
    Parser.Lex();
    return tryParseExpression(Operands, /*IsHash=*/true);

  case AsmToken::At: {
    // 8051 indirect addressing. Forms:
    //   @Ri    -> '@' token followed by an Ri register
    //   @dptr  -> single '@dptr' token
    //   @a+pc / @a+dptr -> '@a' token, then '+' handled as a separate token.
    SMLoc Loc = Parser.getTok().getLoc();
    StringRef Next = getLexer().peekTok().getString();
    if (Next.equals_insensitive("dptr")) {
      Operands.push_back(MCS51Operand::CreateToken("@dptr", Loc));
      Parser.Lex(); // Eat '@'.
      Parser.Lex(); // Eat 'dptr'.
      return false;
    }
    if (Next.equals_insensitive("a")) {
      Operands.push_back(MCS51Operand::CreateToken("@a", Loc));
      Parser.Lex(); // Eat '@'.
      Parser.Lex(); // Eat 'a'.
      return false;
    }
    // '@' followed by a register (R0/R1).
    Operands.push_back(
        MCS51Operand::CreateToken("@", Parser.getTok().getLoc()));
    Parser.Lex(); // Eat '@'.
    return tryParseRegisterOperand(Operands);
  }

  case AsmToken::Slash:
    // 8051 bit-complement prefix: "/bit".
    Operands.push_back(
        MCS51Operand::CreateToken("/", Parser.getTok().getLoc()));
    Parser.Lex(); // Eat '/'.
    return false;

  case AsmToken::Identifier: {
    // The 8051 carry flag ("c"/"cy") and the A/B pair ("ab") are fixed
    // tokens, never symbols.
    StringRef Id = getLexer().getTok().getString();
    if (Id.equals_insensitive("c") || Id.equals_insensitive("cy")) {
      Operands.push_back(
          MCS51Operand::CreateToken("c", Parser.getTok().getLoc()));
      Parser.Lex();
      return false;
    }
    if (Id.equals_insensitive("ab")) {
      Operands.push_back(
          MCS51Operand::CreateToken("ab", Parser.getTok().getLoc()));
      Parser.Lex();
      return false;
    }
    if (Id.equals_insensitive("pc")) {
      Operands.push_back(
          MCS51Operand::CreateToken("pc", Parser.getTok().getLoc()));
      Parser.Lex();
      return false;
    }
    // Try to parse a register, fall through to the next case if it fails.
    if (maybeReg && !tryParseRegisterOperand(Operands)) {
      return false;
    }
    [[fallthrough]];
  }
  case AsmToken::LParen:
  case AsmToken::Integer:
  case AsmToken::Dot:
    return tryParseExpression(Operands);
  case AsmToken::Plus:
  case AsmToken::Minus: {
    // If the sign preceeds a number, parse the number,
    // otherwise treat the sign a an independent token.
    switch (getLexer().peekTok().getKind()) {
    case AsmToken::Integer:
    case AsmToken::BigNum:
    case AsmToken::Identifier:
    case AsmToken::Real:
      if (!tryParseExpression(Operands))
        return false;
      break;
    default:
      break;
    }
    // Treat the token as an independent token.
    Operands.push_back(MCS51Operand::CreateToken(Parser.getTok().getString(),
                                               Parser.getTok().getLoc()));
    Parser.Lex(); // Eat the token.
    return false;
  }
  }

  // Could not parse operand
  return true;
}

ParseStatus MCS51AsmParser::parseMemriOperand(OperandVector &Operands) {
  LLVM_DEBUG(dbgs() << "parseMemriOperand()\n");

  SMLoc E, S;
  MCExpr const *Expression;
  int RegNo;

  // Parse register.
  {
    RegNo = parseRegister();

    if (RegNo == MCS51::NoRegister)
      return ParseStatus::Failure;

    S = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    Parser.Lex(); // Eat register token.
  }

  // Parse immediate;
  {
    if (getParser().parseExpression(Expression))
      return ParseStatus::Failure;

    E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  }

  Operands.push_back(MCS51Operand::CreateMemri(RegNo, Expression, S, E));

  return ParseStatus::Success;
}

bool MCS51AsmParser::parseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                 SMLoc &EndLoc) {
  StartLoc = Parser.getTok().getLoc();
  Reg = parseRegister(/*RestoreOnFailure=*/false);
  EndLoc = Parser.getTok().getLoc();

  return Reg == MCS51::NoRegister;
}

ParseStatus MCS51AsmParser::tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                           SMLoc &EndLoc) {
  StartLoc = Parser.getTok().getLoc();
  Reg = parseRegister(/*RestoreOnFailure=*/true);
  EndLoc = Parser.getTok().getLoc();

  if (Reg == MCS51::NoRegister)
    return ParseStatus::NoMatch;
  return ParseStatus::Success;
}

void MCS51AsmParser::eatComma() {
  if (getLexer().is(AsmToken::Comma)) {
    Parser.Lex();
  } else {
    // GCC allows commas to be omitted.
  }
}

bool MCS51AsmParser::parseInstruction(ParseInstructionInfo &Info,
                                      StringRef Mnemonic, SMLoc NameLoc,
                                      OperandVector &Operands) {
  // Phase-0 bridge: keep only the mnemonic remaps that still require parser-
  // side handling. Simple jump/call/condition aliases are defined in TableGen
  // InstAlias entries.
  StringRef CanonicalMnemonic = Mnemonic;

  Operands.push_back(MCS51Operand::CreateToken(CanonicalMnemonic, NameLoc));

  int OperandNum = -1;
  while (getLexer().isNot(AsmToken::EndOfStatement)) {
    OperandNum++;
    if (OperandNum > 0)
      eatComma();

    ParseStatus ParseRes = MatchOperandParserImpl(Operands, Mnemonic);

    if (ParseRes.isSuccess())
      continue;

    if (ParseRes.isFailure()) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();

      return Error(Loc, "failed to parse register and immediate pair");
    }

    // These specific operands should be treated as addresses/symbols/labels,
    // other than registers.
    bool maybeReg = true;
    if (OperandNum == 1) {
      std::array<StringRef, 8> Insts = {"lds", "adiw", "sbiw", "ldi"};
      for (auto Inst : Insts) {
        if (Inst == Mnemonic) {
          maybeReg = false;
          break;
        }
      }
    } else if (OperandNum == 0) {
      std::array<StringRef, 8> Insts = {"sts", "call", "rcall", "rjmp", "jmp"};
      for (auto Inst : Insts) {
        if (Inst == Mnemonic) {
          maybeReg = false;
          break;
        }
      }
    }

    if (parseOperand(Operands, maybeReg)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");
    }
  }

  // Resolve 8051 bit-address spellings.
  //   - 'c'/'cy' are parsed as the fixed 'c' token (for setb c, clr c, ...).
  //   - flag names and 'base.bit' dotted forms are parsed as symbols and are
  //     resolved here to absolute bit addresses.
  //   - jb/jnb/jbc accept 'c' as the carry bit (0xD7).
  auto MatchBitBase = [](StringRef Base, StringRef Suffix,
                         int64_t &BitAddr) -> bool {
    // Named bit suffixes first.
    bool Named = false;
    if (Base.equals_insensitive("psw")) {
      Named = true;
      if (Suffix.equals_insensitive("p") || Suffix.equals_insensitive("parity"))
        BitAddr = 0xD0;
      else if (Suffix.equals_insensitive("ov") ||
               Suffix.equals_insensitive("ovf") ||
               Suffix.equals_insensitive("overflow"))
        BitAddr = 0xD2;
      else if (Suffix.equals_insensitive("rs0"))
        BitAddr = 0xD3;
      else if (Suffix.equals_insensitive("rs1"))
        BitAddr = 0xD4;
      else if (Suffix.equals_insensitive("f0"))
        BitAddr = 0xD5;
      else if (Suffix.equals_insensitive("ac") ||
               Suffix.equals_insensitive("auxcarry"))
        BitAddr = 0xD6;
      else if (Suffix.equals_insensitive("c") || Suffix.equals_insensitive("cy") ||
               Suffix.equals_insensitive("carry"))
        BitAddr = 0xD7;
      else
        Named = false;
    } else if (Base.equals_insensitive("tcon")) {
      static constexpr const char *Names[] = {"it0", "ie0", "it1", "ie1",
                                              "tr0", "tf0", "tr1", "tf1"};
      for (unsigned I = 0; I < 8; ++I)
        if (Suffix.equals_insensitive(Names[I])) {
          BitAddr = 0x88 + I;
          Named = true;
          break;
        }
    } else if (Base.equals_insensitive("scon")) {
      static constexpr const char *Names[] = {"ri", "ti", "rb8", "tb8",
                                              "ren", "sm2", "sm1", "sm0"};
      for (unsigned I = 0; I < 8; ++I)
        if (Suffix.equals_insensitive(Names[I])) {
          BitAddr = 0x98 + I;
          Named = true;
          break;
        }
    } else if (Base.equals_insensitive("ie")) {
      Named = true;
      if (Suffix.equals_insensitive("ex0"))
        BitAddr = 0xA8;
      else if (Suffix.equals_insensitive("et0"))
        BitAddr = 0xA9;
      else if (Suffix.equals_insensitive("ex1"))
        BitAddr = 0xAA;
      else if (Suffix.equals_insensitive("et1"))
        BitAddr = 0xAB;
      else if (Suffix.equals_insensitive("es"))
        BitAddr = 0xAC;
      else if (Suffix.equals_insensitive("et2"))
        BitAddr = 0xAD;
      else if (Suffix.equals_insensitive("ea"))
        BitAddr = 0xAF;
      else
        Named = false;
    } else if (Base.equals_insensitive("ip")) {
      Named = true;
      if (Suffix.equals_insensitive("px0"))
        BitAddr = 0xB8;
      else if (Suffix.equals_insensitive("pt0"))
        BitAddr = 0xB9;
      else if (Suffix.equals_insensitive("px1"))
        BitAddr = 0xBA;
      else if (Suffix.equals_insensitive("pt1"))
        BitAddr = 0xBB;
      else if (Suffix.equals_insensitive("ps"))
        BitAddr = 0xBC;
      else if (Suffix.equals_insensitive("pt2"))
        BitAddr = 0xBD;
      else
        Named = false;
    }

    if (Named)
      return true;

    // Numeric bit suffix for every bit-addressable SFR byte.
    uint8_t Addr = 0;
    if (Base.equals_insensitive("p0"))
      Addr = 0x80;
    else if (Base.equals_insensitive("tcon"))
      Addr = 0x88;
    else if (Base.equals_insensitive("p1"))
      Addr = 0x90;
    else if (Base.equals_insensitive("scon"))
      Addr = 0x98;
    else if (Base.equals_insensitive("p2"))
      Addr = 0xA0;
    else if (Base.equals_insensitive("ie"))
      Addr = 0xA8;
    else if (Base.equals_insensitive("p3"))
      Addr = 0xB0;
    else if (Base.equals_insensitive("ip"))
      Addr = 0xB8;
    else if (Base.equals_insensitive("psw"))
      Addr = 0xD0;
    else if (Base.equals_insensitive("acc"))
      Addr = 0xE0;
    else if (Base.equals_insensitive("b"))
      Addr = 0xF0;
    else
      return false;

    int64_t BitNo = -1;
    if (!Suffix.getAsInteger(10, BitNo) && BitNo >= 0 && BitNo <= 7) {
      BitAddr = Addr + BitNo;
      return true;
    }
    return false;
  };

  auto ResolveBitAddress = [this, &MatchBitBase](MCS51Operand &Op) -> bool {
    if (!Op.isImm())
      return false;
    const auto *SRE = dyn_cast<MCSymbolRefExpr>(Op.getImm());
    if (!SRE)
      return false;
    StringRef Name = SRE->getSymbol().getName();

    int64_t BitAddr = -1;
    // Bare flag names.
    if (Name.equals_insensitive("carry"))
      BitAddr = 0xD7;
    else if (Name.equals_insensitive("auxcarry") || Name.equals_insensitive("ac"))
      BitAddr = 0xD6;
    else if (Name.equals_insensitive("f0"))
      BitAddr = 0xD5;
    else if (Name.equals_insensitive("rs1"))
      BitAddr = 0xD4;
    else if (Name.equals_insensitive("rs0"))
      BitAddr = 0xD3;
    else if (Name.equals_insensitive("overflow") || Name.equals_insensitive("ov"))
      BitAddr = 0xD2;
    else if (Name.equals_insensitive("parity") || Name.equals_insensitive("p"))
      BitAddr = 0xD0;
    else {
      size_t Dot = Name.rfind('.');
      if (Dot != StringRef::npos && Dot != 0 && Dot + 1 < Name.size()) {
        StringRef Base = Name.substr(0, Dot);
        StringRef Suffix = Name.substr(Dot + 1);
        MatchBitBase(Base, Suffix, BitAddr);
      }
    }

    if (BitAddr < 0)
      return false;
    Op.makeImm(MCConstantExpr::create(BitAddr, getContext()));
    return true;
  };

  StringRef Mn = static_cast<MCS51Operand &>(*Operands[0]).getToken();
  bool IsBitBranch = Mn.equals_insensitive("jb") ||
                     Mn.equals_insensitive("jnb") ||
                     Mn.equals_insensitive("jbc");

  for (unsigned I = 1; I < Operands.size(); ++I) {
    MCS51Operand &Op = static_cast<MCS51Operand &>(*Operands[I]);
    if (Op.isToken()) {
      // 'c' is the carry bit (0xD7) when used as a jb/jnb/jbc operand.
      if (IsBitBranch && Op.getToken().equals_insensitive("c"))
        Op.makeImm(MCConstantExpr::create(0xD7, getContext()));
      continue;
    }

    // A dotted spelling that cannot be resolved to a bit address is an error
    // rather than a symbol reference. Symbols that begin with '.' (temporary
    // labels and the current-location symbol) are not bit addresses.
    if (Op.isImm()) {
      if (const auto *SRE = dyn_cast<MCSymbolRefExpr>(Op.getImm())) {
        StringRef Name = SRE->getSymbol().getName();
        if (!Name.empty() && Name[0] == '.')
          continue;
        if (Name.contains('.') && !ResolveBitAddress(Op))
          return Error(Op.getStartLoc(), "invalid bit address");
      }
    }

    ResolveBitAddress(Op);
  }

  Parser.Lex(); // Consume the EndOfStatement
  return false;
}

ParseStatus MCS51AsmParser::parseDirective(llvm::AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();
  if (IDVal.lower() == ".long")
    return parseLiteralValues(SIZE_LONG, DirectiveID.getLoc());
  if (IDVal.lower() == ".word" || IDVal.lower() == ".short")
    return parseLiteralValues(SIZE_WORD, DirectiveID.getLoc());
  if (IDVal.lower() == ".byte")
    return parseLiteralValues(1, DirectiveID.getLoc());
  return ParseStatus::NoMatch;
}

ParseStatus MCS51AsmParser::parseLiteralValues(unsigned SizeInBytes, SMLoc L) {
  MCAsmParser &Parser = getParser();
  MCS51MCELFStreamer &MCS51Streamer =
      static_cast<MCS51MCELFStreamer &>(Parser.getStreamer());
  AsmToken Tokens[2];
  size_t ReadCount = Parser.getLexer().peekTokens(Tokens);
  if (ReadCount == 2 && Parser.getTok().getKind() == AsmToken::Identifier &&
      Tokens[0].getKind() == AsmToken::Minus &&
      Tokens[1].getKind() == AsmToken::Identifier) {
    MCSymbol *Symbol = getContext().getOrCreateSymbol(".text");
    MCS51Streamer.emitValueForModiferKind(Symbol, SizeInBytes, L,
                                        MCS51MCExpr::VK_MCS51_None);
    return ParseStatus::NoMatch;
  }

  if (Parser.getTok().getKind() == AsmToken::Identifier &&
      Parser.getLexer().peekTok().getKind() == AsmToken::LParen) {
    StringRef ModifierName = Parser.getTok().getString();
    MCS51MCExpr::VariantKind ModifierKind =
        MCS51MCExpr::getKindByName(ModifierName);
    if (ModifierKind != MCS51MCExpr::VK_MCS51_None) {
      Parser.Lex();
      Parser.Lex(); // Eat the modifier and parenthesis
    } else {
      return Error(Parser.getTok().getLoc(), "unknown modifier");
    }
    MCSymbol *Symbol =
        getContext().getOrCreateSymbol(Parser.getTok().getString());
    MCS51Streamer.emitValueForModiferKind(Symbol, SizeInBytes, L, ModifierKind);
    Lex(); // Eat the symbol name.
    if (parseToken(AsmToken::RParen))
      return ParseStatus::Failure;
    return parseEOL();
  }

  auto parseOne = [&]() -> bool {
    const MCExpr *Value;
    if (Parser.parseExpression(Value))
      return true;
    Parser.getStreamer().emitValue(Value, SizeInBytes, L);
    return false;
  };
  return (parseMany(parseOne));
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeMCS51AsmParser() {
  RegisterMCAsmParser<MCS51AsmParser> X(getTheMCS51Target());
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "MCS51GenAsmMatcher.inc"

// Uses enums defined in MCS51GenAsmMatcher.inc
unsigned MCS51AsmParser::validateTargetOperandClass(MCParsedAsmOperand &AsmOp,
                                                  unsigned ExpectedKind) {
  MCS51Operand &Op = static_cast<MCS51Operand &>(AsmOp);
  MatchClassKind Expected = static_cast<MatchClassKind>(ExpectedKind);

  // NOTE: Unlike AVR, bare numbers in 8051 assembly are direct addresses,
  // never register numbers. Do not reinterpret immediates as registers.

  if (Op.isReg()) {
    // If the instruction uses a register pair but we got a single, lower
    // register we perform a "class cast".
    if (isSubclass(Expected, MCK_DREGS)) {
      unsigned correspondingDREG = toDREG(Op.getReg());

      if (correspondingDREG != MCS51::NoRegister) {
        Op.makeReg(correspondingDREG);
        return validateOperandClass(Op, Expected, *STI);
      }
    }
  }
  return Match_InvalidOperand;
}
